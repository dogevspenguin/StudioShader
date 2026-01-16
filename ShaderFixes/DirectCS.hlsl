// --- DirectCS.hlsl ---
// Precomputes Direct Light Color (Transmittance) for clouds
// X-Axis: Sun Zenith Angle (approx -1 to 1)
// Y-Axis: Altitude (Ground to CLOUD_LIMIT) 
// Output: RGB = Transmittance Color, A = Visibility Factor

RWBuffer<float4> DirectLightBuffer : register(u0);

// --- CONSTANTS ---
static const float PI = 3.14159265359;
static const float R_PLANET = 6360000.0;
static const float R_ATM = 6460000.0; // 100km Atmosphere

// FIXED: Reduced to 20km to focus resolution on Cloud Layer (Troposphere)
// This prevents sampling empty space (40km+) which returns pure white.
// IMPORTANT: Update your PS sampling divisor to 20000.0!
static const float MAX_CLOUD_ALT = 20000.0; 

// Solar Angular Radius (approx 0.26 degrees in radians)
static const float SUN_RADIUS_RAD = 0.00465;

// Atmosphere Coefficients
static const float fac = 10.0;
static const float3 BETA_R = float3(5.8e-6, 1.35e-5, 3.31e-5); // Normalized Rayleigh
static const float3 BETA_M_EXT = float3(4.4e-7, 4.4e-7, 4.4e-7) * fac;
static const float3 BETA_ABS_OZONE = float3(6.50e-7, 1.881e-6, 8.5e-8);

// Scale Heights
static const float H_RAYLEIGH = 8000.0;
static const float H_MIE = 1200.0;

// --- INTERSECTION HELPER ---
float RaySphereIntersect(float3 ro, float3 rd, float rad) {
    float b = dot(ro, rd);
    float c = dot(ro, ro) - rad * rad;
    if (c > 0.0f && b > 0.0) return -1.0;
    float discr = b * b - c;
    if (discr < 0.0) return -1.0;
    
    // We want the forward intersection.
    // Since we are usually inside R_ATM, t1 is behind us (-ve) and t2 is forward (+ve).
    // If we are checking R_PLANET (Ground), we might be outside.
    
    float t1 = -b - sqrt(discr);
    float t2 = -b + sqrt(discr);
    
    if (t2 < 0.0) return -1.0;
    if (t1 > 0.0) return t1; // We are outside, hitting the front
    return t2; // We are inside, hitting the exit
}

// --- MAIN KERNEL ---
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint width = 32;
    uint height = 32;
    
    if (id.x >= width || id.y >= height) return;
    int bufferIndex = id.y * 32 + id.x;
    
    // 1. Map Coordinates
    float2 uv = (float2(id.xy) + 0.5) / float2(width, height);
    
    // Y-Axis: Altitude 0 to 20km
    float altitude = uv.y * MAX_CLOUD_ALT; 
    
    float3 P = float3(0, R_PLANET + altitude, 0);

    // Sun Angle: Map X to Cosine of Zenith [-1, 1]
    // -1 = Nadir (Down), 0 = Horizon, 1 = Zenith (Up)
    float cosZenith = (uv.x - 0.5) * 2.0;
    float sinZenith = sqrt(saturate(1.0 - cosZenith * cosZenith));
    
    // 2. Solar Disk Integration
    float3 accumulatedTransmittance = float3(0, 0, 0);
    float accumulatedVisibility = 0.0;
    
    const int DISK_SAMPLES = 8;
    
    [loop]
    for (int s = 0; s < DISK_SAMPLES; s++) {
        // Jitter angle for soft sun edges
        float sampleProgress = (float(s) / float(DISK_SAMPLES - 1)) * 2.0 - 1.0;
        float angleOffset = sampleProgress * SUN_RADIUS_RAD;
        
        float currentAngle = acos(cosZenith) + angleOffset;
        float3 sampleDir = float3(sin(currentAngle), cos(currentAngle), 0.0);
        
        // 3. Occlusion Check (Planet)
        float distToGround = RaySphereIntersect(P, sampleDir, R_PLANET);
        if (distToGround > 0.0) {
             // Blocked by planet: Adds 0 transmittance, 0 visibility
             continue;
        }
        
        // 4. Atmosphere Ray Marching
        float distToSpace = RaySphereIntersect(P, sampleDir, R_ATM);
        if (distToSpace <= 0.0) {
            accumulatedTransmittance += float3(1, 1, 1);
            accumulatedVisibility += 1.0;
            continue;
        }

        // Increased steps for better sunset gradients
        const int STEP_COUNT = 64; 
        float stepSize = distToSpace / float(STEP_COUNT);
        float3 opticalDepth = float3(0, 0, 0);
        
        // Midpoint Integration
        [loop]
        for (int i = 0; i < STEP_COUNT; i++) {
            float t = (float(i) + 0.5) * stepSize;
            float3 currPos = P + sampleDir * t;
            
            float h = length(currPos) - R_PLANET;
            
            // Densities
            float dR = exp(-h / H_RAYLEIGH);
            float dM = exp(-h / H_MIE);
            float dO = max(0.0, 1.0 - abs(h - 25000.0) / 15000.0);
            
            opticalDepth += float3(dR, dM, dO) * stepSize;
        }
        
        // Calculate Transmittance
        float3 tau = BETA_R * opticalDepth.x + BETA_M_EXT * opticalDepth.y + BETA_ABS_OZONE * opticalDepth.z;
        float3 transmittance = exp(-tau);
        
        accumulatedTransmittance += transmittance;
        accumulatedVisibility += 1.0;
    }

    // 5. Average and Store
    float3 finalTransmittance = float3(0,0,0);
    
    // FIX: Avoid division by zero if all samples are blocked (NaN => White)
    if (accumulatedVisibility > 0.001) {
        finalTransmittance = accumulatedTransmittance / accumulatedVisibility;
    } 
    // If blocked, finalTransmittance remains 0 (Black)

    DirectLightBuffer[bufferIndex] = float4(finalTransmittance, 1.0);
}