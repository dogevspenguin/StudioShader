// --- MultiScatCS.hlsl ---
RWBuffer<float4> MultiScatBuffer : register(u0);



static const float PI = 3.14159265359;
static const float R_PLANET = 6360000.0;
static const float R_ATM = 6460000.0;
static const float fac = 10.0;
static const float3 BETA_R = float3(5.8e-6, 1.35e-5, 3.31e-5);
static const float3 BETA_M_SCAT = float3(4.0e-7, 4.0e-7, 4.0e-7)*fac;
static const float3 BETA_M_EXT = float3(4.4e-7, 4.4e-7, 4.4e-7)*fac;
static const float3 BETA_ABS_OZONE = float3(6.50e-7, 1.881e-6, 8.5e-8);

float RaySphereIntersect(float3 ro, float3 rd, float rad) {
    float b = dot(ro, rd);
    float c = dot(ro, ro) - rad * rad;
    if (c > 0.0f && b > 0.0) return -1.0;
    float discr = b * b - c;
    if (discr < 0.0) return -1.0;
    return -b + sqrt(discr);
}

float3 GetFibonacciSphere(int i, int n) {
    float phi = 2.0 * PI * (float(i) / 1.61803398875);
    float z = 1.0 - (2.0 * float(i) + 1.0) / float(n);
    float r = sqrt(saturate(1.0 - z * z));
    return float3(r * cos(phi), r * sin(phi), z);
}

[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    // FIX 1: Update Resolution
    float width = 128.0; 
    float height = 128.0;
    
    int bufferIndex = id.y * 128 + id.x; // Update index stride
    float2 uv = (float2(id.xy) + 0.5) / float2(width, height);

    float cosSunZenith = (uv.x - 0.5) * 2.0;
    float3 sunDir = float3(sqrt(saturate(1.0 - cosSunZenith * cosSunZenith)), cosSunZenith, 0);
    float heightPoint = R_PLANET + (uv.y * (R_ATM - R_PLANET));
    float3 worldPos = float3(0, heightPoint, 0);

    float3 L_2nd = float3(0,0,0);
    float3 f_ms = float3(0,0,0);
    const int SAMPLE_COUNT = 128;

    for (int i = 0; i < SAMPLE_COUNT; i++) {
        float3 rayDir = GetFibonacciSphere(i, SAMPLE_COUNT);
        
        if (RaySphereIntersect(worldPos, rayDir, R_PLANET) > 0.0) continue; 
        float atmDist = RaySphereIntersect(worldPos, rayDir, R_ATM);
        if (atmDist <= 0.0) continue;

        float stepSize = atmDist / 15.0;
        float3 throughput = 1.0;
        
        for (int j = 0; j < 15; j++) {
            float3 p = worldPos + rayDir * ((j + 0.5) * stepSize);
            float h = length(p) - R_PLANET;
            float ozoneDensity = max(0.0, 1.0 - abs(h - 25000.0) / 15000.0);

            float3 sigma_s = BETA_R * exp(-h/8000.0) + BETA_M_SCAT * exp(-h/1200.0);
            float3 sigma_t = BETA_R * exp(-h/8000.0) + BETA_M_EXT * exp(-h/1200.0) + BETA_ABS_OZONE * ozoneDensity;
            
            float3 T_step = exp(-sigma_t * stepSize);
            float3 T_to_P = throughput * exp(-sigma_t * (stepSize * 0.5));
            
            f_ms += sigma_s * T_to_P * stepSize;

            // FIX 2: Check Sun Visibility Properly
            float distToSunSpace = RaySphereIntersect(p, sunDir, R_ATM);
            float earthBlock = RaySphereIntersect(p, sunDir, R_PLANET);

            // Only add light if not blocked by Earth
            if (earthBlock < 0.0 && distToSunSpace > 0.0) {
                 // --- FIX 3: CALCULATE T_SUN (The "Orange" Fix) ---
                 // We must check how much atmosphere the sun passes through to reach 'p'.
                 // A simple 4-step loop is enough for this approximation.
                 float3 T_sun = float3(1,1,1);
                 float sunStep = distToSunSpace / 4.0;
                 for(int k=0; k<4; k++) {
                     float3 pSun = p + sunDir * ((float(k)+0.5) * sunStep);
                     float hSun = length(pSun) - R_PLANET;
		     float ozoneSun = max(0.0, 1.0 - abs(hSun - 25000.0) / 15000.0);
                     float3 density = BETA_R * exp(-hSun/8000.0) + BETA_M_EXT * exp(-hSun/1200.0)+BETA_ABS_OZONE * ozoneSun;
                     T_sun *= exp(-density * sunStep);
                 }

                 L_2nd += sigma_s * T_to_P * T_sun * stepSize;
            }
            
            throughput *= T_step;
        }
    }

    f_ms /= float(SAMPLE_COUNT);
    L_2nd /= float(SAMPLE_COUNT);

    float3 F_ms = 1.0 / (1.0 - f_ms);
    float3 Psi_ms = L_2nd * F_ms;

    MultiScatBuffer[bufferIndex] = float4(Psi_ms, 1.0);

}
