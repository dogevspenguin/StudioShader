RWBuffer<float4> AerialPerspectiveLUT : register(u0);
Buffer<float4> MultiScatBuffer : register(t120);
Buffer<float4> CameraData : register(t10);

// --------------------------------------------------------------------------
// TUNED CONSTANTS (Matches baf016 / Hillaire 2020)
// --------------------------------------------------------------------------
static const float scale = 3.28084;
static const float PI = 3.14159265359;
static const float R_PLANET = 6360000.0 * scale;
static const float R_ATM = 6460000.0 * scale;

// SCALE HEIGHTS
static const float H_RAYLEIGH = 8000.0 * scale;
static const float H_MIE = 3000.0 * scale;

// COEFFICIENTS
static const float facr = 1.0;
static const float3 BETA_R_SCAT = float3(5.8e-6, 1.35e-5, 3.31e-5) * facr;

static const float fac = 10.0;
static const float3 BETA_M_SCAT = float3(4.0e-7, 4.0e-7, 4.0e-7) * fac;
static const float3 BETA_M_EXT  = float3(4.4e-7, 4.4e-7, 4.4e-7) * fac;

static const float3 BETA_ABS_OZONE = float3(6.50e-7, 1.881e-6, 8.5e-8) * facr;

// --------------------------------------------------------------------------
// HELPER FUNCTIONS
// --------------------------------------------------------------------------

float3 GetMultiScattering(float3 p, float3 sunDir) {
    float h = length(p) - R_PLANET;
    float cosSun = dot(normalize(p), sunDir);
    
    float u = 0.5 + 0.5 * cosSun;
    float v = saturate(h / (R_ATM - R_PLANET));
    
    float width = 32.0; float height = 32.0;
    float x = u * width - 0.5; float y = v * height - 0.5;
    int x0 = clamp(int(floor(x)), 0, 31); int y0 = clamp(int(floor(y)), 0, 31);
    int x1 = clamp(x0 + 1, 0, 31); int y1 = clamp(y0 + 1, 0, 31);
    float wx = frac(x); float wy = frac(y);
    
    float3 v00 = MultiScatBuffer.Load(y0 * 32 + x0).rgb;
    float3 v10 = MultiScatBuffer.Load(y0 * 32 + x1).rgb;
    float3 v01 = MultiScatBuffer.Load(y1 * 32 + x0).rgb;
    float3 v11 = MultiScatBuffer.Load(y1 * 32 + x1).rgb;
    
    return lerp(lerp(v00, v10, wx), lerp(v01, v11, wx), wy);
}

float PhaseRayleigh(float cosTheta) { 
    return (3.0 / (16.0 * PI)) * (1.0 + cosTheta * cosTheta); 
}

float PhaseMie(float cosTheta) {
    float a = 1.5;
    float gamma = acos(clamp(cosTheta, -1.0, 1.0));
    float num = 2.0 * (1.0 + a * a);
    float den = 2.0 * PI * (1.0 + exp(-a * PI));
    float term1 = (1.0 - exp(-a * PI * 0.5)) * exp(-a * gamma);
    float term2 = exp(-a * PI * 0.5) * exp(-a * (PI - gamma));
    return (num / den) * (term1 + term2);
}

float2 ResolveSphereIntersection(float3 rayOrigin, float3 rayDir, float sphereRadius) {
    float a = dot(rayDir, rayDir);
    float b = 2.0 * dot(rayDir, rayOrigin);
    float c = dot(rayOrigin, rayOrigin) - (sphereRadius * sphereRadius);
    float d = b * b - 4.0 * a * c;
    if (d < 0.0) return float2(-1.0, -1.0);
    float sqrtD = sqrt(d);
    return float2((-b - sqrtD) / (2.0 * a), (-b + sqrtD) / (2.0 * a));
}

float3 GetOpticalDepth(float3 p, float3 lightDir) {
    float2 planetIntersect = ResolveSphereIntersection(p, lightDir, R_PLANET);
    if (planetIntersect.y > 0.0) return float3(1e9, 1e9, 1e9); 
    
    float2 intersect = ResolveSphereIntersection(p, lightDir, R_ATM);
    float distToSpace = intersect.y;
    if (distToSpace < 0.0) return float3(0,0,0);
    
    int steps = 16;
    float stepSize = distToSpace / float(steps);
    float stepSizeMeters = stepSize / scale; 
    float odR = 0.0; float odM = 0.0; float odO = 0.0;
    
    for(int i = 0; i < steps; i++) {
        float3 curr = p + lightDir * (float(i) * stepSize + 0.5 * stepSize);
        float h = max(0.0, length(curr) - R_PLANET);
        float ozoneDensity = max(0.0, 1.0 - abs(h - 25000.0) / 15000.0);
        odR += exp(-h / H_RAYLEIGH) * stepSizeMeters;
        odM += exp(-h / H_MIE) * stepSizeMeters;
        odO += ozoneDensity * stepSizeMeters;
    }
    return float3(odR, odM, odO);
}

// --------------------------------------------------------------------------
// MAIN KERNEL
// --------------------------------------------------------------------------
[numthreads(8, 8, 1)] 
void main(uint3 id : SV_DispatchThreadID) {
    uint width = 32; uint height = 32; uint depth = 100;
    if (id.x >= width || id.y >= height || id.z >= depth) return;

    // 1. Camera & Sun Setup
    float3 camView = CameraData[0].xyz;
    float3 camPos = CameraData[1].xyz;
    float3 camRight = CameraData[2].xyz;
    float3 camUp = CameraData[3].xyz;
    float4 fovData = CameraData[6];
    if (length(fovData.xy) < 0.001) fovData = float4(1.0, 1.0, 1.0, 1.0); 

    float3 sunDir = -normalize(CameraData[4].xyz);
    float3 sunColor = CameraData[5].xyz;
    float lampBrightness = length(sunColor);
    float rawSunY = CameraData[4].y;
    
    if (rawSunY > 0.2 && lampBrightness < 2.0) sunColor = float3(0,0,0);
    else sunColor = float3(3.0, 3.0, 3.0);
    float3 sunIntensity = sunColor;

    // 2. Ray Reconstruction (Standard)
    float2 uv = (float2(id.xy) + 0.5) / float2(width, height);
    float2 ndc = uv * 2.0 - 1.0;
    float2 screenRay = ndc * fovData.zw; 
    float3 r1 = camUp * screenRay.y;
    float3 rayVec = (camRight * screenRay.x) - r1; 
    rayVec = -camView + rayVec; 
    float3 viewDir = normalize(rayVec);

    // 3. Setup Geometry & Slicing
    float camAltitude = max(10.0, camPos.y);
    float3 startPos = float3(0, R_PLANET + camAltitude, 0);

    float2 atmHit = ResolveSphereIntersection(startPos, viewDir, R_ATM);
    float2 planetHit = ResolveSphereIntersection(startPos, viewDir, R_PLANET);
    
    float marchStart = 0.0;
    float maxAtmDist = atmHit.y;
    
    if (camAltitude > (R_ATM - R_PLANET)) { // In Space
        marchStart = atmHit.x;
        maxAtmDist = atmHit.y;
    }
    if (planetHit.x > 0.0) maxAtmDist = min(maxAtmDist, planetHit.x);

    // --- DEPTH SLICING (Hillaire 2020) ---
    // 32 slices mapped to 32km (or limited by atmosphere boundary)
    float sliceStep = float(id.z + 1) / float(depth); // 0..1
    // Squared distribution for better near-field resolution
    float targetDist = (sliceStep * sliceStep) * (100000.0 * scale); 
    
    // Clamp to valid atmosphere geometry
    float marchEnd = min(maxAtmDist, targetDist);
    
    // If the slice is invalid (e.g. looking away from atm in space), return
    if (marchEnd < marchStart) {
         uint vIdx = id.z * (width * height) + id.y * width + id.x;
         AerialPerspectiveLUT[vIdx * 2] = float4(0,0,0,1);
         return;
    }

    // 4. Ray Marching
    int numSteps = 16;
    float stepSize = (marchEnd - marchStart) / float(numSteps);
    float stepSizeMeters = stepSize / scale; 

    // Accumulators
    float3 totalInScat = float3(0,0,0);
    float3 totalTransmittance = float3(1,1,1); // Start at 1.0 (Full transmission)

    // Current Optical Depth from Camera
    float odR_View = 0.0;
    float odM_View = 0.0;
    float odO_View = 0.0;
    
    float3 BETA_R_EXT = BETA_R_SCAT; 
    
    float cosThetaSun = dot(viewDir, sunDir);
    float pR = PhaseRayleigh(cosThetaSun);
    float pM = PhaseMie(cosThetaSun);

    for (int i = 0; i < numSteps; i++) {
        float dist = marchStart + float(i) * stepSize + 0.5 * stepSize;
        float3 p = startPos + viewDir * dist;
        float h = max(0.0, length(p) - R_PLANET);
        
        // --- 3-Channel Transmittance Calculation ---
        float ozoneDensity = max(0.0, 1.0 - abs(h - 25000.0) / 15000.0);
        float densityR = exp(-h / H_RAYLEIGH);
        float densityM = exp(-h / H_MIE);
        
        // Step Optical Depths
        float dR = densityR * stepSizeMeters;
        float dM = densityM * stepSizeMeters;
        float dO = ozoneDensity * stepSizeMeters;
        
        // Accumulate View Optical Depth
        odR_View += dR;
        odM_View += dM;
        odO_View += dO;
        
        // Calculate View Transmittance (Beer-Lambert Law)
        // T = exp(-(BetaR*OdR + BetaM*OdM + BetaO*OdO))
        float3 extinction = BETA_R_EXT * odR_View + BETA_M_EXT * odM_View + BETA_ABS_OZONE * odO_View;
        float3 T_View = exp(-extinction);
        
        // Store the final transmittance for this slice
        totalTransmittance = T_View;
        
        // --- In-Scattering ---
        // 1. Sun Shadow
        float3 odSun = GetOpticalDepth(p, sunDir);
        float shadow = (odSun.x > 1e8) ? 0.0 : 1.0;
        float3 T_Sun = float3(0,0,0);
        
        if (shadow > 0.0) {
            T_Sun = exp(-(BETA_R_EXT * odSun.x + BETA_M_EXT * odSun.y + BETA_ABS_OZONE * odSun.z));
        }
        
        // 2. Single Scattering (Rayleigh + Mie)
        // L_scat = (BetaR * PhaseR + BetaM * PhaseM) * SunIntensity * T_Sun * T_View * StepSize
        float3 scatteringS = (BETA_R_SCAT * densityR * pR + BETA_M_SCAT * densityM * pM) * sunIntensity;
        float3 lightS = scatteringS * T_Sun * T_View * stepSizeMeters;
        
        // 3. Multi Scattering (Isotropic approximation from Hillaire)
        float3 psiMS = GetMultiScattering(p, sunDir);
        float3 scatteringMS = (BETA_R_SCAT * densityR + BETA_M_SCAT * densityM) * 1.0; // 1.0 = Isotropic Phase
        float3 lightMS = scatteringMS * psiMS * sunIntensity * T_View * stepSizeMeters;
        
        totalInScat += (lightS + lightMS);
    }
    
    // 5. Tone Map (Only for Visualization)
    // In a real pipeline, AP LUTs usually store raw HDR values.
    float exposure = 0.5; 
    float3 col = totalInScat; 

    // 6. Write Output
    uint voxelIndex = id.z * (width * height) + id.y * width + id.x;
    AerialPerspectiveLUT[voxelIndex * 2] = float4(col, 1.0);
    // Optionally write Transmittance to the second slot if you want to inspect it later
    AerialPerspectiveLUT[voxelIndex * 2 + 1] = float4(totalTransmittance, 1.0);
    //AerialPerspectiveLUT[voxelIndex * 2 + 1] = float4(1.0, 1.0, 1.0, 1.0);
}