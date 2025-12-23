// FIX: Use RWBuffer<float> (Single component)
RWBuffer<float> Output : register(u0);

[numthreads(1, 1, 1)]
void main()
{
    // 1. Read the current time (Allowed because it is a single float)
    float currentTime = Output[0];

    // 2. Increment it safely
    // If it's broken (NaN or 0), reset to 1.0
    if (currentTime <= 0.0) currentTime = 1.0;

    // 3. Write it back
    Output[0] = currentTime + 0.005;
}