// ---- Created with 3Dmigoto v1.3.16 ----
// Custom Atmosphere Shader - Unit Corrected (1m = 3.048 units)

cbuffer CB0 : register(b0)
{
  struct
  {
    row_major float4x4 ViewProjection[2];
    float4 ViewRight;
    float4 ViewUp;
    float4 ViewDir;
    float3 CameraPosition[2];
    float3 AmbientColor;
    float4 SkyAmbient;
    float3 Lamp0Color; // PRIMARY LIGHT
    float3 Lamp0Dir;
    float3 Lamp1Color;
    float4 FogParams;
    float4 FogColor_GlobalForceFieldTime;
    float4 Exposure_DoFDistance;
    float4 LightConfig0;
    float4 LightConfig1;
    float4 LightConfig2;
    float4 LightConfig3;
    float4 ShadowMatrix0;
    float4 ShadowMatrix1;
    float4 ShadowMatrix2;
    float4 RefractionBias_FadeDistance_GlowFactor_Free;
    float4 TextureData_ShadowInfo;
    float4 SkyGradientTop_EnvDiffuse;
    float4 SkyGradientBottom_EnvSpec;
    float4 AmbientColorNoIBL_CubeBlend;
    float4 SkyAmbientNoIBL;
    float4 AmbientCube[12];
    float4 CascadeSphere0;
    float4 CascadeSphere1;
    float4 CascadeSphere2;
    float4 CascadeSphere3;
    float2 invViewportWH;
    float2 viewportScale;
    float debugAuthLodMode;
    float padding;
    float hqDist;
    float localLightDist;
    float sunDist;
    float hybridLerpSlope;
    float evsmPosExp;
    float evsmNegExp;
    float globalShadow;
    float shadowBias;
    float packedAlphaRef;
    float debugFlags;
    row_major float4x4 FroxelTransform;
    float4 SkyboxRotation0;
    float4 SkyboxRotation1;
    float4 SkyboxRotation2;
  } CB0 : packoffset(c0);
}

cbuffer CB4 : register(b4)
{
  struct
  {
    float4 FoV;
    float2 shearOffset;
    float2 pad;
  } CB4 : packoffset(c0);
}

SamplerState DiffuseMapSampler_s : register(s0);
Texture2D<float4> DiffuseMapTexture : register(t0);

// --------------------------------------------------------------------------
// TUNED CONSTANTS (Units: 1 meter = 3.048 game units)
// --------------------------------------------------------------------------
static const float scale = 3.28084; 
static const float PI = 3.14159265359;

// GEOMETRY (In Game Units)
static const float R_PLANET = 6360000.0 * scale;
static const float R_ATM = 6420000.0 * scale;

// SCALE HEIGHTS (Converted to Game Units so the atmosphere isn't "compressed")
static const float H_RAYLEIGH = 8000.0 * scale; 
static const float H_MIE = 1200.0 * scale;

// BASE COEFFICIENTS (Physically based, per meter)
// We do NOT multiply these by scale. We convert the distance to meters instead.
static const float3 BETA_R_SCAT = float3(5.8e-6, 1.35e-5, 3.31e-5); 

// Mie Scat: Low value to prevent "White Wall" look
static const float3 BETA_M_SCAT = float3(4.0e-7, 4.0e-7, 4.0e-7); 
// Mie Ext: Slightly higher than scattering to ensure haze has shadow/body
static const float3 BETA_M_EXT  = float3(4.4e-7, 4.4e-7, 4.4e-7); 

// --------------------------------------------------------------------------
// HELPER FUNCTIONS
// --------------------------------------------------------------------------

float2 ResolveSphereIntersection(float3 rayOrigin, float3 rayDir, float sphereRadius) {
    float a = dot(rayDir, rayDir);
    float b = 2.0 * dot(rayDir, rayOrigin);
    float c = dot(rayOrigin, rayOrigin) - (sphereRadius * sphereRadius);
    float d = b * b - 4.0 * a * c;

    if (d < 0.0) return float2(-1.0, -1.0);
    float sqrtD = sqrt(d);
    return float2((-b - sqrtD) / (2.0 * a), (-b + sqrtD) / (2.0 * a));
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

float2 GetOpticalDepth(float3 p, float3 lightDir) {
    // Self-Shadowing
    float2 planetIntersect = ResolveSphereIntersection(p, lightDir, R_PLANET);
    if (planetIntersect.y > 0.0) return float2(1e9, 1e9); 

    float2 intersect = ResolveSphereIntersection(p, lightDir, R_ATM);
    float distToSpace = intersect.y; // This is in Game Units
    
    if (distToSpace < 0.0) return float2(0,0);

    int steps = 4;
    float stepSize = distToSpace / float(steps);
    
    // UNIT FIX 1: Convert step size from Units to Meters for the integration
    float stepSizeMeters = stepSize / scale; 

    float odR = 0.0;
    float odM = 0.0;

    for(int i = 0; i < steps; i++) {
        float3 curr = p + lightDir * (float(i) * stepSize + 0.5 * stepSize);
        float h = max(0.0, length(curr) - R_PLANET); // Altitude in Units
        
        // UNIT FIX 2: h is Units, H_RAYLEIGH is Units. The ratio is correct.
        odR += exp(-h / H_RAYLEIGH) * stepSizeMeters;
        odM += exp(-h / H_MIE) * stepSizeMeters;
    }
    return float2(odR, odM);
}

// --------------------------------------------------------------------------
// MAIN SHADER
// --------------------------------------------------------------------------
void main(
  float4 v0 : SV_Position0,
  float2 v1 : TEXCOORD0,
  float4 v2 : COLOR0,
  out float4 o0 : SV_Target0)
{
    // --- 1. RAY RECONSTRUCTION ---
    float4 r0;
    float3 r1;
    r0.xy = float2(0.5, 0.5) * v0.xy;
    r0.xy = r0.xy / CB0.viewportScale.xy;
    r0.zw = r0.xy * CB4.FoV.xy + float2(-1, -1);
    r0.zw = CB4.FoV.zw * r0.zw + CB4.shearOffset.xy;
    r1.xyz = CB0.ViewUp.xyz * r0.www;
    r1.xyz = r0.zzz * CB0.ViewRight.xyz + -r1.xyz;
    r1.xyz = -CB0.ViewDir.xyz + r1.xyz;
    float3 viewDir = normalize(r1);

    // --- 2. SETUP LIGHTS & SUN/MOON DISCRIMINATION ---
    float3 sunDir = -normalize(CB0.Lamp0Dir.xyz);
    float lampBrightness = length(CB0.Lamp0Color.xyz);
    
    bool isHighAltitude = sunDir.y > 0.2; 
    bool isDimLight = lampBrightness < 2.0;
    
    float3 sunIntensity;

    if (isDimLight && isHighAltitude) {
        // Night / Moon -> Black
        sunIntensity = float3(0.0, 0.0, 0.0);
    } 
    else {
        // Day / Sunset -> Bright
        sunIntensity = float3(20.0, 20.0, 20.0);
    }
    
    // --- 3. DYNAMIC EXTINCTION ---
    float sunHeight = saturate(sunDir.y);
    float betaMult = lerp(1.0, 0.0, smoothstep(0.0, 0.15, sunHeight));
    float3 BETA_R_EXT = BETA_R_SCAT * betaMult;

    // --- 4. SCENE SETUP ---
    float camAltitude = max(10.0, CB0.CameraPosition[0].y);
    float3 startPos = float3(0, R_PLANET + camAltitude, 0);

    float2 atmHit = ResolveSphereIntersection(startPos, viewDir, R_ATM);
    float2 planetHit = ResolveSphereIntersection(startPos, viewDir, R_PLANET);

    float marchStart = 0.0;
    float marchEnd = atmHit.y;
    
    bool inSpace = camAltitude > (R_ATM - R_PLANET);

    if (inSpace) {
        if (atmHit.x < 0.0 && atmHit.y < 0.0) {
            float cosSun = dot(viewDir, sunDir);
            float sunDisk = smoothstep(0.9998, 0.99995, cosSun);
            o0 = float4(sunIntensity * sunDisk, 1.0);
            return;
        }
        marchStart = atmHit.x;
        marchEnd = atmHit.y;
    }

    if (planetHit.x > 0.0) {
        marchEnd = planetHit.x;
    }
    
    if (marchEnd < marchStart) {
        o0 = float4(0,0,0,1);
        return;
    }

    // --- 5. RAY MARCHING ---
    int numSteps = 16;
    float stepSize = (marchEnd - marchStart) / float(numSteps);
    
    // UNIT FIX 3: Convert the step size to Meters for calculation
    float stepSizeMeters = stepSize / scale;

    float3 totalRayleighSun = float3(0,0,0);
    float3 totalMieSun = float3(0,0,0);
    
    float odR_View = 0.0;
    float odM_View = 0.0;

    for (int i = 0; i < numSteps; i++) {
        float dist = marchStart + float(i) * stepSize + 0.5 * stepSize;
        float3 p = startPos + viewDir * dist;
        float h = max(0.0, length(p) - R_PLANET);
        
        // Density Integration
        // h is in Units, H_RAYLEIGH is in Units -> Ratio is correct.
        // We multiply by stepSizeMeters because Beta is 1/Meters.
        float hr = exp(-h / H_RAYLEIGH) * stepSizeMeters;
        float hm = exp(-h / H_MIE) * stepSizeMeters;
        
        odR_View += hr;
        odM_View += hm;
        
        float2 odSun = GetOpticalDepth(p, sunDir);
        float shadowSun = (odSun.x > 1e8) ? 0.0 : 1.0;
        
        if (shadowSun > 0.0) {
            float3 tauSun = BETA_R_EXT * (odR_View + odSun.x) + 
                            BETA_M_EXT * (odM_View + odSun.y);
            float3 attnSun = exp(-tauSun);
            totalRayleighSun += hr * attnSun;
            totalMieSun += hm * attnSun;
        }
    }

    // --- 6. PHASE & COMBINATION ---
    float cosThetaSun = dot(viewDir, sunDir);
    float pR_Sun = PhaseRayleigh(cosThetaSun);
    float pM_Sun = PhaseMie(cosThetaSun);

    float3 atmSun = sunIntensity * (totalRayleighSun * BETA_R_SCAT * pR_Sun + totalMieSun * BETA_M_SCAT * pM_Sun);

    // --- 7. DISK RENDERING ---
    float sunDiskVal = smoothstep(0.9998, 0.99995, cosThetaSun);
    float2 camToSunOD = GetOpticalDepth(startPos, sunDir);
    float3 sunExtinction = exp(-(BETA_R_EXT * camToSunOD.x + BETA_M_EXT * camToSunOD.y));
    float3 sunColor = sunIntensity * sunDiskVal * sunExtinction * 10.0;

    float3 finalColor = atmSun + sunColor;

    // --- 8. TONE MAPPING ---
    float exposure = max(0.2, CB0.Exposure_DoFDistance.y);
    float3 col = finalColor * exposure;
    col = col / (1.0 + col);
    o0.xyz = pow(saturate(col), 1.0/2.2); 
    o0.w = 1.0;
}
