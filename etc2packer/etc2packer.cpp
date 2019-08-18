// This is a simple example application for using CVTT's ETC kernels to compress ETC textures.
// It only compresses a single texture level.

#include <string.h>
#include <algorithm>

#include "stb_image/stb_image.h"

#include "ktxheader.h"
#include "etc2packer.h"
#include "../ConvectionKernels.h"

static void *allocshim(void *context, size_t size)
{
    return _aligned_malloc(size, 16);
}

static void freeshim(void *context, void *ptr, size_t size)
{
    _aligned_free(ptr);
}

enum TargetFormat
{
    ETC1,
    ETC2_RGB,
    ETC2_RGBA,
    ETC2_Punchthrough,
    R11_Unsigned,
    R11_Signed,
};

const char *g_formatNames[] =
{
    "etc1",
    "etc2rgb",
    "etc2rgba",
    "etc2punchthrough",
    "r11u",
    "r11s",
};

void PrintUsageAndExit()
{
    fprintf(stderr, "Usage: etc2packer [options] input output\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "-format <format> - Selects output format.  Format is one of:\n");
    fprintf(stderr, "   etc1 - ETC1\n");
    fprintf(stderr, "   etc2rgb - ETC2 RGB\n");
    fprintf(stderr, "   etc2rgba - ETC2 RGBA\n");
    fprintf(stderr, "   etc2punchthrough - ETC2 RGB with punchthrough alpha\n");
    fprintf(stderr, "-fakebt709 - Use fake BT.709 error metric (same as etc2comp, significantly slower)\n");
    fprintf(stderr, "-uniform - Use uniform color weights (overrides -fakebt709)\n");
    exit(-1);
}

int main(int argc, const char **argv)
{
    TargetFormat targetFormat = ETC2_RGB;
    bool useFakeBT709 = false;
    bool useUniform = false;

    const char *inputPath = NULL;
    const char *outputPath = NULL;

    if (argc < 3)
        PrintUsageAndExit();

    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "-format"))
        {
            i++;
            if (i == argc)
                PrintUsageAndExit();

            bool foundFormat = false;
            for (int f = 0; f < sizeof(g_formatNames) / sizeof(g_formatNames[0]); f++)
            {
                if (!strcmp(argv[i], g_formatNames[f]))
                {
                    targetFormat = static_cast<TargetFormat>(f);
                    foundFormat = true;
                    break;
                }
            }
        }
        else if (!strcmp(argv[i], "-fakebt709"))
        {
            useFakeBT709 = true;
        }
        else if (!strcmp(argv[i], "-uniform"))
        {
            useUniform = true;
        }
        else
        {
            if (i != argc - 2)
                PrintUsageAndExit();

            inputPath = argv[i];
            outputPath = argv[i + 1];
            break;
        }
    }

    int w, h, channels;
    stbi_uc *image = stbi_load(inputPath, &w, &h, &channels, 4);

    if (!image)
    {
        fprintf(stderr, "Could not load input image\n");
        return -1;
    }

    static const uint8_t ktxIdentifier[12] =
    {
        0xAB, 0x4B, 0x54, 0x58, // first four bytes of Byte[12] identifier
        0x20, 0x31, 0x31, 0xBB, // next four bytes of Byte[12] identifier
        0x0D, 0x0A, 0x1A, 0x0A  // final four bytes of Byte[12] identifier
    };

	KtxHeader_t ktxHeader;
	memcpy(ktxHeader.identifier, ktxIdentifier, 12);

	ktxHeader.endianness = 0x04030201;
	ktxHeader.glType = 0;
	ktxHeader.glTypeSize = 1;
	ktxHeader.glFormat = 0;

	ktxHeader.glInternalFormat = (unsigned int)KtxHeader_t::InternalFormat::ETC2_RGB8;
	ktxHeader.glBaseInternalFormat = (unsigned int)KtxHeader_t::BaseInternalFormat::ETC2_RGB8;

	ktxHeader.pixelWidth = w;
	ktxHeader.pixelHeight = h;
	ktxHeader.pixelDepth = 0;
	ktxHeader.numberOfArrayElements = 0;
	ktxHeader.numberOfFaces = 0;
	ktxHeader.bytesOfKeyValueData = 0;

	ktxHeader.pixelDepth = 0;
	ktxHeader.numberOfArrayElements = 0;
	ktxHeader.numberOfFaces = 1;
	ktxHeader.numberOfMipmapLevels = 1;

    unsigned int blockSizeBytes = 8;

    switch (targetFormat)
    {
    case ETC1:
        ktxHeader.glInternalFormat = (unsigned int)KtxHeader_t::InternalFormat::ETC1_RGB8;
        ktxHeader.glBaseInternalFormat = (unsigned int)KtxHeader_t::BaseInternalFormat::ETC1_RGB8;
        blockSizeBytes = 8;
        break;
    case ETC2_RGB:
        ktxHeader.glInternalFormat = (unsigned int)KtxHeader_t::InternalFormat::ETC2_RGB8;
        ktxHeader.glBaseInternalFormat = (unsigned int)KtxHeader_t::BaseInternalFormat::ETC2_RGB8;
        blockSizeBytes = 8;
        break;
    case ETC2_RGBA:
        ktxHeader.glInternalFormat = (unsigned int)KtxHeader_t::InternalFormat::ETC2_RGBA8;
        ktxHeader.glBaseInternalFormat = (unsigned int)KtxHeader_t::BaseInternalFormat::ETC2_RGBA8;
        blockSizeBytes = 16;
        break;
    case ETC2_Punchthrough:
        ktxHeader.glInternalFormat = (unsigned int)KtxHeader_t::InternalFormat::ETC2_RGB8A1;
        ktxHeader.glBaseInternalFormat = (unsigned int)KtxHeader_t::BaseInternalFormat::ETC2_RGB8A1;
        blockSizeBytes = 8;
        break;
    case R11_Unsigned:
        ktxHeader.glInternalFormat = (unsigned int)KtxHeader_t::InternalFormat::ETC2_R11;
        ktxHeader.glBaseInternalFormat = (unsigned int)KtxHeader_t::BaseInternalFormat::ETC2_R11;
        blockSizeBytes = 8;
        break;
    case R11_Signed:
        ktxHeader.glInternalFormat = (unsigned int)KtxHeader_t::InternalFormat::ETC2_SIGNED_R11;
        ktxHeader.glBaseInternalFormat = (unsigned int)KtxHeader_t::BaseInternalFormat::ETC2_R11;
        blockSizeBytes = 8;
        break;
    }

    uint8_t alphaOutputBlock[8 * cvtt::NumParallelBlocks];
    uint8_t outputBlock[8 * cvtt::NumParallelBlocks];

	FILE *f = fopen(outputPath, "wb");
    if (!f)
    {
        fprintf(stderr, "Could not open output file\n");
        return -1;
    }

    int blockWidth = (w + 3) / 4;
    int blockHeight = (h + 3) / 4;

	fwrite(&ktxHeader, sizeof(ktxHeader), 1, f);
	uint32_t dataSize = blockWidth * blockHeight * blockSizeBytes;
	fwrite(&dataSize, 4, 1, f);

    cvtt::Options options;

    if (useUniform)
        options.flags |= cvtt::Flags::Uniform;
    else if (useFakeBT709)
        options.flags |= cvtt::Flags::ETC_UseFakeBT709;

    cvtt::ETC1CompressionData* compressionData1 = NULL;
    cvtt::ETC2CompressionData* compressionData2 = NULL;

    if (targetFormat == ETC1)
        compressionData1 = cvtt::Kernels::AllocETC1Data(allocshim, nullptr);

    if (targetFormat == ETC2_RGB || targetFormat == ETC2_RGBA || targetFormat == ETC2_Punchthrough)
        compressionData2 = cvtt::Kernels::AllocETC2Data(allocshim, nullptr, options);

	for (int y = 0; y < h; y += 4)
	{
        cvtt::PixelBlockU8 pixelBlocks[8];
        cvtt::PixelBlockScalarS16 pixelBlockSigned[8];
        cvtt::PixelBlockScalarS16 pixelBlockUnsigned[8];
        for (int x = 0; x < w; x += 32)
		{
            for (int block = 0; block < cvtt::NumParallelBlocks; block++)
            {
                for (int subY = 0; subY < 4; subY++)
                {
                    int clampedY = std::min(y + subY, h - 1);

                    const uint8_t *inputRow = image + (clampedY) * w * 4;
                    for (int subX = 0; subX < 4; subX++)
                    {
                        int clampedX = std::min(x + subX + block * 4, w - 1);

                        int rgba[4];
                        for (int ch = 0; ch < 4; ch++)
                            rgba[ch] = inputRow[clampedX * 4 + ch];

                        for (int ch = 0; ch < 4; ch++)
                            pixelBlocks[block].m_pixels[subY * 4 + subX][ch] = rgba[ch];

                        double rgbaTotal = rgba[0] + rgba[1] + rgba[2];
                        double normalizedUnsigned = rgbaTotal / (255.0 * 3.0);
                        double normalizedSigned = normalizedUnsigned * 2.0 - 1.0;

                        pixelBlockUnsigned[block].m_pixels[subY * 4 + subX] = static_cast<int>(floor(normalizedUnsigned * 2047.0 + 0.5));
                        pixelBlockSigned[block].m_pixels[subY * 4 + subX] = static_cast<int>(floor(normalizedUnsigned * 1023.0 + 0.5));
                    }
                }
            }

            if (targetFormat == ETC2_RGBA)
                cvtt::Kernels::EncodeETC2Alpha(alphaOutputBlock, pixelBlocks, options);

            switch (targetFormat)
            {
            case ETC1:
                cvtt::Kernels::EncodeETC1(outputBlock, pixelBlocks, options, compressionData1);
                break;
            case R11_Unsigned:
                cvtt::Kernels::EncodeETC2Alpha11(outputBlock, pixelBlockUnsigned, false, options);
                break;
            case R11_Signed:
                cvtt::Kernels::EncodeETC2Alpha11(outputBlock, pixelBlockSigned, true, options);
                break;
            case ETC2_Punchthrough:
                cvtt::Kernels::EncodeETC2PunchthroughAlpha(outputBlock, pixelBlocks, options, compressionData2);
                break;
            case ETC2_RGB:
            case ETC2_RGBA:
                cvtt::Kernels::EncodeETC2(outputBlock, pixelBlocks, options, compressionData2);
                break;
            }

            int writableBlocks = std::min<int>(cvtt::NumParallelBlocks, (w - x + 3) / 4);

            for (int block = 0; block < writableBlocks; block++)
            {
                if (targetFormat == ETC2_RGBA)
                    fwrite(alphaOutputBlock + block * 8, 8, 1, f);
                fwrite(outputBlock + block * 8, 8, 1, f);
            }
        }
	}

    if (compressionData1)
        cvtt::Kernels::ReleaseETC1Data(compressionData1, freeshim);

    if (compressionData2)
        cvtt::Kernels::ReleaseETC2Data(compressionData2, freeshim);

	stbi_image_free(image);

	return 0;
}
