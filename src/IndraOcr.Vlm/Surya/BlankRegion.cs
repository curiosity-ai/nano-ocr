using IndraOcr.Vlm.Imaging;

namespace IndraOcr.Vlm.Surya;

/// <summary><c>surya/common/blank.py</c>: blank or near-uniform image regions.</summary>
public static class BlankRegion
{
    /// <summary><c>BLANK_WHITE_THRESHOLD</c>.</summary>
    public const int WhiteThreshold = 245;

    /// <summary><c>BLANK_PIXEL_FRACTION</c>.</summary>
    public const double PixelFraction = 0.99;

    /// <summary><c>UNIFORM_COLOR_STD</c>.</summary>
    public const double UniformColorStd = 8.0;

    /// <summary><c>near_white_fraction</c>.</summary>
    public static double NearWhiteFraction(RgbImage image, int whiteThreshold = WhiteThreshold)
    {
        ReadOnlySpan<byte> p = image.Pixels;
        int n = image.Width * image.Height;
        if (n == 0)
        {
            return 0;
        }

        long white = 0;
        for (int i = 0; i < n; i++)
        {
            if (p[i * 3] >= whiteThreshold && p[(i * 3) + 1] >= whiteThreshold && p[(i * 3) + 2] >= whiteThreshold)
            {
                white++;
            }
        }

        return white / (double)n;
    }

    /// <summary>
    /// <c>is_blank_region</c>: mostly near-white, or every channel's standard deviation (numpy's,
    /// population, in float64) below <paramref name="uniformColorStd"/>. An empty crop is not blank.
    /// </summary>
    public static bool IsBlank(
        RgbImage image,
        int whiteThreshold = WhiteThreshold,
        double blankPixelFraction = PixelFraction,
        double uniformColorStd = UniformColorStd)
    {
        int n = image.Width * image.Height;
        if (n == 0)
        {
            return false;
        }

        if (NearWhiteFraction(image, whiteThreshold) > blankPixelFraction)
        {
            return true;
        }

        ReadOnlySpan<byte> p = image.Pixels;
        double maxStd = 0;
        for (int c = 0; c < 3; c++)
        {
            double sum = 0;
            for (int i = 0; i < n; i++)
            {
                sum += p[(i * 3) + c];
            }

            double mean = sum / n;
            double squares = 0;
            for (int i = 0; i < n; i++)
            {
                double d = p[(i * 3) + c] - mean;
                squares += d * d;
            }

            maxStd = Math.Max(maxStd, Math.Sqrt(squares / n));
        }

        return maxStd < uniformColorStd;
    }
}
