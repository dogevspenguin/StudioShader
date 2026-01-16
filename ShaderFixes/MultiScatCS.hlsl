// --- MultiScatCS.hlsl ---
RWBuffer<float4> MultiScatBuffer : register(u0);

static const float PI = 3.14159265359;
static const float R_PLANET = 6360000.0;
static const float R_ATM = 6460000.0;
static const float fac = 10; // Your tuned factor
static const float3 BETA_R = float3(5.8e-6, 1.35e-5, 3.31e-5)*1;
static const float3 BETA_M_SCAT = float3(4.0e-7, 4.0e-7, 4.1e-7)*fac;
static const float3 BETA_M_EXT = float3(4.4e-7, 4.4e-7, 4.4e-7)*fac;
static const float3 BETA_ABS_OZONE = float3(6.50e-7, 1.881e-6, 8.5e-8)*1;

// Returns distance to the Exit point (for Atmosphere)
float RayAtmosphereIntersect(float3 ro, float3 rd, float rad) {
    float b = dot(ro, rd);
    float c = dot(ro, ro) - rad * rad;
    float discr = b * b - c;
    if (discr < 0.0) return -1.0;
    // We want the far intersection (exit)
    return -b + sqrt(discr);
}

// Returns distance to the Entry point (for Planet)
float RayPlanetIntersect(float3 ro, float3 rd, float rad) {
    float b = dot(ro, rd);
    float c = dot(ro, ro) - rad * rad;
    if (c > 0.0 && b > 0.0) return -1.0; // Outside and looking away
    float discr = b * b - c;
    if (discr < 0.0) return -1.0;
    
    // We want the NEAR intersection (entry)
    // If we are inside (c < 0), this might be negative, but we are assuming worldPos is > R_PLANET
    return -b - sqrt(discr);
}

float3 GetFibonacciSphere(int i, int n) {
    float phi = 2.0 * PI * (float(i) / 1.61803398875);
    float z = 1.0 - (2.0 * float(i) + 1.0) / float(n);
    float r = sqrt(saturate(1.0 - z * z));
    return float3(r * cos(phi), r * sin(phi), z);
}

[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    float width = 32.0;
    float height = 32.0;
    
    int bufferIndex = id.y * 32 + id.x;
    float2 uv = (float2(id.xy) + 0.5) / float2(width, height);

    // Map UV to World Position & Sun Direction
    float cosSunZenith = (uv.x - 0.5) * 2.0;
    float3 sunDir = float3(sqrt(saturate(1.0 - cosSunZenith * cosSunZenith)), cosSunZenith, 0);
    
    // Altitude mapping (0 to Top of Atmosphere)
    float heightPoint = R_PLANET + (uv.y * (R_ATM - R_PLANET));
    float3 worldPos = float3(0, heightPoint, 0);

    float3 L_2nd = float3(0,0,0);
    float3 f_ms = float3(0,0,0);
    
    const int SAMPLE_COUNT = 64; // 64 is usually enough for fibonacci
    
    [loop]
    for (int i = 0; i < SAMPLE_COUNT; i++) {
        float3 rayDir = GetFibonacciSphere(i, SAMPLE_COUNT);
        
        // --- KEY FIX STARTS HERE ---
        
        // 1. Get distance to Atmosphere Exit
        float atmDist = RayAtmosphereIntersect(worldPos, rayDir, R_ATM);
        
        // 2. Get distance to Planet Entry
        float earthDist = RayPlanetIntersect(worldPos, rayDir, R_PLANET);
        
        // 3. Determine actual ray length
        float maxDist = atmDist;
        if (earthDist > 0.0) {
            // If we hit the planet, the ray stops THERE.
            // We do NOT discard the ray ("continue"). The air up to this point counts!
            maxDist = earthDist;
        }
        
        if (maxDist <= 0.0) continue; // Safety check

        float stepSize = maxDist / 15.0;
        float3 throughput = 1.0;
        
        [loop]
        for (int j = 0; j < 20; j++) {
            float3 p = worldPos + rayDir * ((j + 0.5) * stepSize);
            float pHeight = length(p);
            float h = pHeight - R_PLANET;

            // Safety clamp
            if (h < 0.0) h = 0.0; 

            float ozoneDensity = max(0.0, 1.0 - abs(h - 25000.0*0.3048) / 15000.0*0.3048);
            float3 sigma_s = BETA_R * exp(-h/8000.0) + BETA_M_SCAT * exp(-h/3000.0);
            float3 sigma_t = BETA_R * exp(-h/8000.0) + BETA_M_EXT * exp(-h/3000.0) + BETA_ABS_OZONE * ozoneDensity;

            // Energy Conserving Integration
            float3 T_step = exp(-sigma_t * stepSize);
            float3 T_next = throughput * T_step;
            float3 S_over_T = sigma_s / max(0.0000001, sigma_t);
            float3 energyTransfer = S_over_T * (throughput - T_next);

            f_ms += energyTransfer;

            // --- L_2nd Calculation (FIX: NO HARD SHADOWS) ---
            // We ignore planet occlusion for the source term to ensure smooth ambient light
            // The density of the atmosphere alone provides the necessary extinction for night.
            float distToSunSpace = RayAtmosphereIntersect(p, sunDir, R_ATM);

            if (distToSunSpace > 0.0) {
                 float3 T_sun = float3(1,1,1);
                 float sunStep = distToSunSpace / 4.0;
                 
                 [unroll]
                 for(int k=0; k<4; k++) {
                     float3 pSun = p + sunDir * ((float(k)+0.5) * sunStep);
                     float hSun = length(pSun) - R_PLANET;
                     
                     // If ray passes through earth, hSun becomes negative.
                     // We must handle this density gracefully to avoid bright spots from "negative height"
                     // Standard fix: Mirror density or Clamp. Hillaire uses Clamp.
                     if (hSun < 0.0) hSun = 0.0; 

                     float ozoneSun = max(0.0, 1.0 - abs(hSun - 25000.0) / 15000.0);
                     float3 density = BETA_R * exp(-hSun/8000.0) + BETA_M_EXT * exp(-hSun/1200.0) + BETA_ABS_OZONE * ozoneSun;
                     T_sun *= exp(-density * sunStep);
                 }
                 L_2nd += energyTransfer * T_sun;
            }
            
            throughput = T_next;
            if (all(throughput < 0.001)) break;
        }
        
        // NOTE: We do NOT add any ground color here.
        // If the ray hit the ground (earthDist > 0), the loop finishes, 
        // and we simply stop adding energy. This acts as a black, absorbing ground.
    }

    f_ms /= float(SAMPLE_COUNT);
    L_2nd /= float(SAMPLE_COUNT);
    
    // Prevent explosion if f_ms is too close to 1.0 (though slice integration handles this well)
    float3 F_ms = 1.0 / (1.0 - min(f_ms, 0.99)); 
    float3 Psi_ms = L_2nd * F_ms;

    MultiScatBuffer[bufferIndex] = float4(Psi_ms, 1.0);
}