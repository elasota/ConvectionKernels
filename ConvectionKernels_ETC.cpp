/*
Convection Texture Tools
Copyright (c) 2018-2019 Eric Lasota

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject
to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

-------------------------------------------------------------------------------------

Portions based on DirectX Texture Library (DirectXTex)

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.

http://go.microsoft.com/fwlink/?LinkId=248926
*/
#include "ConvectionKernels_Config.h"

#if !defined(CVTT_SINGLE_FILE) || defined(CVTT_SINGLE_FILE_IMPL)

#include "ConvectionKernels.h"
#include "ConvectionKernels_ETC.h"
#include "ConvectionKernels_ETC1.h"
#include "ConvectionKernels_ETC2.h"
#include "ConvectionKernels_ETC2_Rounding.h"
#include "ConvectionKernels_ParallelMath.h"
#include "ConvectionKernels_FakeBT709_Rounding.h"

cvtt::ParallelMath::Float cvtt::Internal::ETCComputer::ComputeErrorUniform(const MUInt15 pixelA[3], const MUInt15 pixelB[3])
{
    MSInt16 d0 = ParallelMath::LosslessCast<MSInt16>::Cast(pixelA[0]) - ParallelMath::LosslessCast<MSInt16>::Cast(pixelB[0]);
    MFloat fd0 = ParallelMath::ToFloat(d0);
    MFloat error = fd0 * fd0;
    for (int ch = 1; ch < 3; ch++)
    {
        MSInt16 d = ParallelMath::LosslessCast<MSInt16>::Cast(pixelA[ch]) - ParallelMath::LosslessCast<MSInt16>::Cast(pixelB[ch]);
        MFloat fd = ParallelMath::ToFloat(d);
        error = error + fd * fd;
    }
    return error;
}

cvtt::ParallelMath::Float cvtt::Internal::ETCComputer::ComputeErrorWeighted(const MUInt15 reconstructed[3], const MFloat preWeightedPixel[3], const Options options)
{
    MFloat dr = ParallelMath::ToFloat(reconstructed[0]) * options.redWeight - preWeightedPixel[0];
    MFloat dg = ParallelMath::ToFloat(reconstructed[1]) * options.greenWeight - preWeightedPixel[1];
    MFloat db = ParallelMath::ToFloat(reconstructed[2]) * options.blueWeight - preWeightedPixel[2];

    return dr * dr + dg * dg + db * db;
}

cvtt::ParallelMath::Float cvtt::Internal::ETCComputer::ComputeErrorFakeBT709(const MUInt15 reconstructed[3], const MFloat preWeightedPixel[3])
{
    MFloat yuv[3];
    ConvertToFakeBT709(yuv, reconstructed);

    MFloat dy = yuv[0] - preWeightedPixel[0];
    MFloat du = yuv[1] - preWeightedPixel[1];
    MFloat dv = yuv[2] - preWeightedPixel[2];

    return dy * dy + du * du + dv * dv;
}

void cvtt::Internal::ETCComputer::TestHalfBlock(MFloat &outError, MUInt16 &outSelectors, MUInt15 quantizedPackedColor, const MUInt15 pixels[8][3], const MFloat preWeightedPixels[8][3], const MSInt16 modifiers[4], bool isDifferential, const Options &options)
{
    MUInt15 quantized[3];
    MUInt15 unquantized[3];

    for (int ch = 0; ch < 3; ch++)
    {
        quantized[ch] = (ParallelMath::RightShift(quantizedPackedColor, (ch * 5)) & ParallelMath::MakeUInt15(31));

        if (isDifferential)
            unquantized[ch] = (quantized[ch] << 3) | ParallelMath::RightShift(quantized[ch], 2);
        else
            unquantized[ch] = (quantized[ch] << 4) | quantized[ch];
    }

    MUInt16 selectors = ParallelMath::MakeUInt16(0);
    MFloat totalError = ParallelMath::MakeFloatZero();

    MUInt15 u15_255 = ParallelMath::MakeUInt15(255);
    MSInt16 s16_zero = ParallelMath::MakeSInt16(0);

    MUInt15 unquantizedModified[4][3];
    for (unsigned int s = 0; s < 4; s++)
        for (int ch = 0; ch < 3; ch++)
            unquantizedModified[s][ch] = ParallelMath::Min(ParallelMath::ToUInt15(ParallelMath::Max(ParallelMath::ToSInt16(unquantized[ch]) + modifiers[s], s16_zero)), u15_255);

    bool isUniform = ((options.flags & cvtt::Flags::Uniform) != 0);
    bool isFakeBT709 = ((options.flags & cvtt::Flags::ETC_UseFakeBT709) != 0);

    for (int px = 0; px < 8; px++)
    {
        MFloat bestError = ParallelMath::MakeFloat(FLT_MAX);
        MUInt16 bestSelector = ParallelMath::MakeUInt16(0);

        for (unsigned int s = 0; s < 4; s++)
        {
            MFloat error;
            if (isFakeBT709)
                error = ComputeErrorFakeBT709(unquantizedModified[s], preWeightedPixels[px]);
            else if (isUniform)
                error = ComputeErrorUniform(pixels[px], unquantizedModified[s]);
            else
                error = ComputeErrorWeighted(unquantizedModified[s], preWeightedPixels[px], options);

            ParallelMath::FloatCompFlag errorBetter = ParallelMath::Less(error, bestError);
            bestSelector = ParallelMath::Select(ParallelMath::FloatFlagToInt16(errorBetter), ParallelMath::MakeUInt16(s), bestSelector);
            bestError = ParallelMath::Min(error, bestError);
        }

        totalError = totalError + bestError;
        selectors = selectors | (bestSelector << (px * 2));
    }

    outError = totalError;
    outSelectors = selectors;
}

cvtt::ParallelMath::Int16CompFlag cvtt::Internal::ETCComputer::ETCDifferentialIsLegalForChannel(const MUInt15 &a, const MUInt15 &b)
{
    MSInt16 diff = ParallelMath::LosslessCast<MSInt16>::Cast(b) - ParallelMath::LosslessCast<MSInt16>::Cast(a);

    return ParallelMath::Less(ParallelMath::MakeSInt16(-5), diff) & ParallelMath::Less(diff, ParallelMath::MakeSInt16(4));
}

cvtt::ParallelMath::Int16CompFlag cvtt::Internal::ETCComputer::ETCDifferentialIsLegal(const MUInt15 &a, const MUInt15 &b)
{
    MUInt15 mask = ParallelMath::MakeUInt15(31);

    return ETCDifferentialIsLegalForChannel(ParallelMath::RightShift(a, 10), ParallelMath::RightShift(b, 10))
        & ETCDifferentialIsLegalForChannel(ParallelMath::RightShift(a, 5) & mask, ParallelMath::RightShift(b, 5) & mask)
        & ETCDifferentialIsLegalForChannel(a & mask, b & mask);
}

bool cvtt::Internal::ETCComputer::ETCDifferentialIsLegalForChannelScalar(const uint16_t &a, const uint16_t &b)
{
    int16_t diff = static_cast<int16_t>(b) - static_cast<int16_t>(a);

    return (-4 <= diff) && (diff <= 3);
}

bool cvtt::Internal::ETCComputer::ETCDifferentialIsLegalScalar(const uint16_t &a, const uint16_t &b)
{
    MUInt15 mask = ParallelMath::MakeUInt15(31);

    return ETCDifferentialIsLegalForChannelScalar((a >> 10), (b >> 10))
        & ETCDifferentialIsLegalForChannelScalar((a >> 5) & 31, (b >> 5) & 31)
        & ETCDifferentialIsLegalForChannelScalar(a & 31, b & 31);
}

void cvtt::Internal::ETCComputer::EncodeTMode(uint8_t *outputBuffer, MFloat &bestError, const ParallelMath::Int16CompFlag isIsolated[16], const MUInt15 pixels[16][3], const MFloat preWeightedPixels[16][3], const Options &options)
{
    bool isUniform = ((options.flags & cvtt::Flags::Uniform) != 0);
    bool isFakeBT709 = ((options.flags & cvtt::Flags::ETC_UseFakeBT709) != 0);

    ParallelMath::Int16CompFlag bestIsThisMode = ParallelMath::MakeBoolInt16(false);

    MUInt15 isolatedTotal[3] = { ParallelMath::MakeUInt15(0), ParallelMath::MakeUInt15(0), ParallelMath::MakeUInt15(0) };
    MUInt15 lineTotal[3] = { ParallelMath::MakeUInt15(0), ParallelMath::MakeUInt15(0), ParallelMath::MakeUInt15(0) };

    MUInt15 numPixelsIsolated = ParallelMath::MakeUInt15(0);

    // To speed this up, we compute line total as the sum, then subtract out isolated
    for (unsigned int px = 0; px < 16; px++)
    {
        for (int ch = 0; ch < 3; ch++)
        {
            isolatedTotal[ch] = isolatedTotal[ch] + ParallelMath::SelectOrZero(isIsolated[px], pixels[px][ch]);
            lineTotal[ch] = lineTotal[ch] + pixels[px][ch];
        }
        numPixelsIsolated = numPixelsIsolated + ParallelMath::SelectOrZero(isIsolated[px], ParallelMath::MakeUInt15(1));
    }

    for (int ch = 0; ch < 3; ch++)
        lineTotal[ch] = lineTotal[ch] - isolatedTotal[ch];

    MUInt15 numPixelsLine = ParallelMath::MakeUInt15(16) - numPixelsIsolated;

    MUInt15 isolatedAverageQuantized[3];
    MUInt15 isolatedAverageTargets[3];
    {
        int divisors[ParallelMath::ParallelSize];
        for (int block = 0; block < ParallelMath::ParallelSize; block++)
            divisors[block] = ParallelMath::Extract(numPixelsIsolated, block) * 34;

        MUInt15 addend = (numPixelsIsolated << 4) | numPixelsIsolated;
        for (int ch = 0; ch < 3; ch++)
        {
            // isolatedAverageQuantized[ch] = (isolatedTotal[ch] * 2 + numPixelsIsolated * 17) / (numPixelsIsolated * 34);

            MUInt15 numerator = isolatedTotal[ch] + isolatedTotal[ch];
            if (!isFakeBT709)
                numerator = numerator + addend;

            for (int block = 0; block < ParallelMath::ParallelSize; block++)
            {
                int divisor = divisors[block];
                if (divisor == 0)
                    ParallelMath::PutUInt15(isolatedAverageQuantized[ch], block, 0);
                else
                    ParallelMath::PutUInt15(isolatedAverageQuantized[ch], block, ParallelMath::Extract(numerator, block) / divisor);
            }

            isolatedAverageTargets[ch] = numerator;
        }
    }

    if (isFakeBT709)
        ResolveTHFakeBT709Rounding(isolatedAverageQuantized, isolatedAverageTargets, numPixelsIsolated);

    MUInt15 isolatedColor[3];
    for (int ch = 0; ch < 3; ch++)
        isolatedColor[ch] = (isolatedAverageQuantized[ch]) | (isolatedAverageQuantized[ch] << 4);

    MFloat isolatedError[16];
    for (int px = 0; px < 16; px++)
    {
        if (isFakeBT709)
            isolatedError[px] = ComputeErrorFakeBT709(isolatedColor, preWeightedPixels[px]);
        else if (isUniform)
            isolatedError[px] = ComputeErrorUniform(pixels[px], isolatedColor);
        else
            isolatedError[px] = ComputeErrorWeighted(isolatedColor, preWeightedPixels[px], options);
    }

    MSInt32 bestSelectors = ParallelMath::MakeSInt32(0);
    MUInt15 bestTable = ParallelMath::MakeUInt15(0);
    MUInt15 bestLineColor = ParallelMath::MakeUInt15(0);

    MSInt16 maxLine = ParallelMath::LosslessCast<MSInt16>::Cast(numPixelsLine);
    MSInt16 minLine = ParallelMath::MakeSInt16(0) - maxLine;

    int16_t clusterMaxLine = 0;
    for (int block = 0; block < ParallelMath::ParallelSize; block++)
    {
        int16_t blockMaxLine = ParallelMath::Extract(maxLine, block);
        if (blockMaxLine > clusterMaxLine)
            clusterMaxLine = blockMaxLine;
    }

    int16_t clusterMinLine = -clusterMaxLine;

    int lineDivisors[ParallelMath::ParallelSize];
    for (int block = 0; block < ParallelMath::ParallelSize; block++)
        lineDivisors[block] = ParallelMath::Extract(numPixelsLine, block) * 34;

    MUInt15 lineAddend = (numPixelsLine << 4) | numPixelsLine;

    for (int table = 0; table < 8; table++)
    {
        int numUniqueColors[ParallelMath::ParallelSize];
        MUInt15 uniqueQuantizedColors[31];

        for (int block = 0; block < ParallelMath::ParallelSize; block++)
            numUniqueColors[block] = 0;

        MUInt15 modifier = ParallelMath::MakeUInt15(cvtt::Tables::ETC2::g_thModifierTable[table]);
        MUInt15 modifierOffset = (modifier + modifier);

        for (int16_t offsetPremultiplier = clusterMinLine; offsetPremultiplier <= clusterMaxLine; offsetPremultiplier++)
        {
            MSInt16 clampedOffsetPremultiplier = ParallelMath::Max(minLine, ParallelMath::Min(maxLine, ParallelMath::MakeSInt16(offsetPremultiplier)));
            MSInt16 modifierAddend = ParallelMath::CompactMultiply(clampedOffsetPremultiplier, modifierOffset);

            MUInt15 quantized[3];
            if (isFakeBT709)
            {
                MUInt15 targets[3];
                for (int ch = 0; ch < 3; ch++)
                {
                    //quantized[ch] = std::min<int16_t>(15, std::max(0, (lineTotal[ch] * 2 + modifierOffset * offsetPremultiplier)) / (numDAIILine * 34));
                    MUInt15 numerator = ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::Max(ParallelMath::MakeSInt16(0), ParallelMath::LosslessCast<MSInt16>::Cast(lineTotal[ch] + lineTotal[ch]) + modifierAddend));
                    MUInt15 divided = ParallelMath::MakeUInt15(0);
                    for (int block = 0; block < ParallelMath::ParallelSize; block++)
                    {
                        int divisor = lineDivisors[block];
                        if (divisor == 0)
                            ParallelMath::PutUInt15(divided, block, 0);
                        else
                            ParallelMath::PutUInt15(divided, block, ParallelMath::Extract(numerator, block) / divisor);
                    }
                    quantized[ch] = ParallelMath::Min(ParallelMath::MakeUInt15(15), divided);
                    targets[ch] = numerator;
                }

                ResolveTHFakeBT709Rounding(quantized, targets, numPixelsLine);
            }
            else
            {
                for (int ch = 0; ch < 3; ch++)
                {
                    //quantized[ch] = std::min<int16_t>(15, std::max(0, (lineTotal[ch] * 2 + numDAIILine * 17 + modifierOffset * offsetPremultiplier)) / (numDAIILine * 34));
                    MUInt15 numerator = ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::Max(ParallelMath::MakeSInt16(0), ParallelMath::LosslessCast<MSInt16>::Cast(lineTotal[ch] + lineTotal[ch] + lineAddend) + modifierAddend));
                    MUInt15 divided = ParallelMath::MakeUInt15(0);
                    for (int block = 0; block < ParallelMath::ParallelSize; block++)
                    {
                        int divisor = lineDivisors[block];
                        if (divisor == 0)
                            ParallelMath::PutUInt15(divided, block, 0);
                        else
                            ParallelMath::PutUInt15(divided, block, ParallelMath::Extract(numerator, block) / divisor);
                    }
                    quantized[ch] = ParallelMath::Min(ParallelMath::MakeUInt15(15), divided);
                }
            }

            MUInt15 packedColor = quantized[0] | (quantized[1] << 5) | (quantized[2] << 10);

            for (int block = 0; block < ParallelMath::ParallelSize; block++)
            {
                uint16_t blockPackedColor = ParallelMath::Extract(packedColor, block);
                if (numUniqueColors[block] == 0 || blockPackedColor != ParallelMath::Extract(uniqueQuantizedColors[numUniqueColors[block] - 1], block))
                    ParallelMath::PutUInt15(uniqueQuantizedColors[numUniqueColors[block]++], block, blockPackedColor);
            }
        }

        // Stripe unfilled unique colors
        int maxUniqueColors = 0;
        for (int block = 0; block < ParallelMath::ParallelSize; block++)
        {
            if (numUniqueColors[block] > maxUniqueColors)
                maxUniqueColors = numUniqueColors[block];
        }

        for (int block = 0; block < ParallelMath::ParallelSize; block++)
        {
            uint16_t fillColor = ParallelMath::Extract(uniqueQuantizedColors[0], block);

            int numUnique = numUniqueColors[block];
            for (int fill = numUnique + 1; fill < maxUniqueColors; fill++)
                ParallelMath::PutUInt15(uniqueQuantizedColors[fill], block, fillColor);
        }

        for (int ci = 0; ci < maxUniqueColors; ci++)
        {
            MUInt15 lineColors[3][3];
            for (int ch = 0; ch < 3; ch++)
            {
                MUInt15 quantizedChannel = (ParallelMath::RightShift(uniqueQuantizedColors[ci], (ch * 5)) & ParallelMath::MakeUInt15(15));

                MUInt15 unquantizedColor = (quantizedChannel << 4) | quantizedChannel;
                lineColors[0][ch] = ParallelMath::Min(ParallelMath::MakeUInt15(255), unquantizedColor + modifier);
                lineColors[1][ch] = unquantizedColor;
                lineColors[2][ch] = ParallelMath::ToUInt15(ParallelMath::Max(ParallelMath::MakeSInt16(0), ParallelMath::LosslessCast<MSInt16>::Cast(unquantizedColor) - ParallelMath::LosslessCast<MSInt16>::Cast(modifier)));
            }

            MSInt32 selectors = ParallelMath::MakeSInt32(0);
            MFloat error = ParallelMath::MakeFloatZero();
            for (int px = 0; px < 16; px++)
            {
                MFloat pixelError = isolatedError[px];

                MUInt15 pixelBestSelector = ParallelMath::MakeUInt15(0);
                for (int i = 0; i < 3; i++)
                {
                    MFloat error = isUniform ? ComputeErrorUniform(lineColors[i], pixels[px]) : ComputeErrorWeighted(lineColors[i], preWeightedPixels[px], options);
                    ParallelMath::FloatCompFlag errorBetter = ParallelMath::Less(error, pixelError);
                    pixelError = ParallelMath::Min(error, pixelError);
                    pixelBestSelector = ParallelMath::Select(ParallelMath::FloatFlagToInt16(errorBetter), ParallelMath::MakeUInt15(i + 1), pixelBestSelector);
                }

                error = error + pixelError;
                selectors = selectors | (ParallelMath::ToInt32(pixelBestSelector) << (px * 2));
            }

            ParallelMath::Int16CompFlag errorBetter = ParallelMath::FloatFlagToInt16(ParallelMath::Less(error, bestError));
            bestError = ParallelMath::Min(error, bestError);

            if (ParallelMath::AnySet(errorBetter))
            {
                ParallelMath::ConditionalSet(bestLineColor, errorBetter, uniqueQuantizedColors[ci]);
                ParallelMath::ConditionalSet(bestSelectors, errorBetter, selectors);
                ParallelMath::ConditionalSet(bestTable, errorBetter, ParallelMath::MakeUInt15(table));
                bestIsThisMode = bestIsThisMode | errorBetter;
            }
        }
    }

    const int selectorOrder[] = { 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15 };

    for (int block = 0; block < ParallelMath::ParallelSize; block++)
    {
        if (ParallelMath::Extract(bestIsThisMode, block))
        {
            uint32_t lowBits = 0;
            uint32_t highBits = 0;

            uint16_t blockBestLineColor = ParallelMath::Extract(bestLineColor, block);
            uint16_t blockIsolatedAverageQuantized[3];

            for (int ch = 0; ch < 3; ch++)
                blockIsolatedAverageQuantized[ch] = ParallelMath::Extract(isolatedAverageQuantized[ch], block);

            uint16_t blockBestTable = ParallelMath::Extract(bestTable, block);
            int32_t blockBestSelectors = ParallelMath::Extract(bestSelectors, block);

            int16_t lineColor[3];
            for (int ch = 0; ch < 3; ch++)
                lineColor[ch] = (blockBestLineColor >> (ch * 5)) & 15;

            int rh = ((blockIsolatedAverageQuantized[0] >> 2) & 3);
            int rl = (blockIsolatedAverageQuantized[0] & 3);

            if (rh + rl < 4)
            {
                // Overflow low
                highBits |= 1 << (58 - 32);
            }
            else
            {
                // Overflow high
                highBits |= 7 << (61 - 32);
            }

            highBits |= rh << (59 - 32);
            highBits |= rl << (56 - 32);
            highBits |= blockIsolatedAverageQuantized[1] << (52 - 32);
            highBits |= blockIsolatedAverageQuantized[2] << (48 - 32);
            highBits |= lineColor[0] << (44 - 32);
            highBits |= lineColor[1] << (40 - 32);
            highBits |= lineColor[2] << (36 - 32);
            highBits |= ((blockBestTable >> 1) & 3) << (34 - 32);
            highBits |= 1 << (33 - 32);
            highBits |= (blockBestTable & 1) << (32 - 32);

            for (int px = 0; px < 16; px++)
            {
                int sel = (blockBestSelectors >> (2 * selectorOrder[px])) & 3;
                if ((sel & 0x1) != 0)
                    lowBits |= (1 << px);
                if ((sel & 0x2) != 0)
                    lowBits |= (1 << (16 + px));
            }

            for (int i = 0; i < 4; i++)
                outputBuffer[block * 8 + i] = (highBits >> (24 - i * 8)) & 0xff;
            for (int i = 0; i < 4; i++)
                outputBuffer[block * 8 + i + 4] = (lowBits >> (24 - i * 8)) & 0xff;
        }
    }
}

void cvtt::Internal::ETCComputer::EncodeHMode(uint8_t *outputBuffer, MFloat &bestError, const ParallelMath::Int16CompFlag groupings[16], const MUInt15 pixels[16][3], HModeEval &he, const MFloat preWeightedPixels[16][3], const Options &options)
{
    bool isUniform = ((options.flags & cvtt::Flags::Uniform) != 0);
    bool isFakeBT709 = ((options.flags & cvtt::Flags::ETC_UseFakeBT709) != 0);

    MUInt15 zero15 = ParallelMath::MakeUInt15(0);

    MUInt15 counts[2] = { zero15, zero15 };

    ParallelMath::Int16CompFlag bestIsThisMode = ParallelMath::MakeBoolInt16(false);

    MUInt15 totals[2][3] =
    {
        { zero15, zero15, zero15 },
        { zero15, zero15, zero15 }
    };

    for (unsigned int px = 0; px < 16; px++)
    {
        for (int ch = 0; ch < 3; ch++)
        {
            totals[0][ch] = totals[0][ch] + pixels[px][ch];
            totals[1][ch] = totals[1][ch] + ParallelMath::SelectOrZero(groupings[px], pixels[px][ch]);
        }
        counts[1] = counts[1] + ParallelMath::SelectOrZero(groupings[px], ParallelMath::MakeUInt15(1));
    }

    for (int ch = 0; ch < 3; ch++)
        totals[0][ch] = totals[0][ch] - totals[1][ch];
    counts[0] = ParallelMath::MakeUInt15(16) - counts[1];

    MUInt16 bestSectorBits = ParallelMath::MakeUInt16(0);
    MUInt16 bestSignBits = ParallelMath::MakeUInt16(0);
    MUInt15 bestColors[2] = { zero15, zero15 };
    MUInt15 bestTable = ParallelMath::MakeUInt15(0);

    for (int table = 0; table < 8; table++)
    {
        MUInt15 numUniqueColors = zero15;

        int modifier = cvtt::Tables::ETC1::g_thModifierTable[table];

        for (int sector = 0; sector < 2; sector++)
        {
            for (int block = 0; block < ParallelMath::ParallelSize; block++)
            {
                int blockNumUniqueColors = 0;
                uint16_t blockUniqueQuantizedColors[31];

                int maxOffsetMultiplier = ParallelMath::Extract(counts[sector], block);
                int minOffsetMultiplier = -maxOffsetMultiplier;

                int modifierOffset = modifier * 2;

                int blockSectorCounts = ParallelMath::Extract(counts[sector], block);
                int blockSectorTotals[3];
                for (int ch = 0; ch < 3; ch++)
                    blockSectorTotals[ch] = ParallelMath::Extract(totals[sector][ch], block);

                for (int offsetPremultiplier = minOffsetMultiplier; offsetPremultiplier <= maxOffsetMultiplier; offsetPremultiplier++)
                {
                    // TODO: This isn't ideal for FakeBT709
                    int16_t quantized[3];
                    for (int ch = 0; ch < 3; ch++)
                    {
                        if (blockSectorCounts == 0)
                            quantized[ch] = 0;
                        else
                            quantized[ch] = std::min<int16_t>(15, std::max<int16_t>(0, (blockSectorTotals[ch] * 2 + blockSectorCounts * 17 + modifierOffset * offsetPremultiplier)) / (blockSectorCounts * 34));
                    }

                    uint16_t packedColor = (quantized[0] << 10) | (quantized[1] << 5) | quantized[2];
                    if (blockNumUniqueColors == 0 || packedColor != blockUniqueQuantizedColors[blockNumUniqueColors - 1])
                    {
                        assert(blockNumUniqueColors < 32);
                        blockUniqueQuantizedColors[blockNumUniqueColors++] = packedColor;
                    }
                }

                ParallelMath::PutUInt15(he.numUniqueColors[sector], block, blockNumUniqueColors);

                int baseIndex = 0;
                if (sector == 1)
                    baseIndex = ParallelMath::Extract(he.numUniqueColors[0], block);

                for (int i = 0; i < blockNumUniqueColors; i++)
                    ParallelMath::PutUInt15(he.uniqueQuantizedColors[baseIndex + i], block, blockUniqueQuantizedColors[i]);
            }
        }

        MUInt15 totalColors = he.numUniqueColors[0] + he.numUniqueColors[1];
        int maxErrorColors = 0;
        for (int block = 0; block < ParallelMath::ParallelSize; block++)
            maxErrorColors = std::max<int>(maxErrorColors, ParallelMath::Extract(totalColors, block));

        for (int block = 0; block < ParallelMath::ParallelSize; block++)
        {
            int lastColor = ParallelMath::Extract(totalColors, block);
            uint16_t stripeColor = ParallelMath::Extract(he.uniqueQuantizedColors[0], block);
            for (int i = lastColor; i < maxErrorColors; i++)
                ParallelMath::PutUInt15(he.uniqueQuantizedColors[i], block, stripeColor);
        }

        for (int ci = 0; ci < maxErrorColors; ci++)
        {
            MUInt15 fifteen = ParallelMath::MakeUInt15(15);
            MUInt15 twoFiftyFive = ParallelMath::MakeUInt15(255);
            MSInt16 zeroS16 = ParallelMath::MakeSInt16(0);

            MUInt15 colors[2][3];
            for (int ch = 0; ch < 3; ch++)
            {
                MUInt15 quantizedChannel = ParallelMath::RightShift(he.uniqueQuantizedColors[ci], ((2 - ch) * 5)) & fifteen;

                MUInt15 unquantizedColor = (quantizedChannel << 4) | quantizedChannel;
                colors[0][ch] = ParallelMath::Min(twoFiftyFive, unquantizedColor + modifier);
                colors[1][ch] = ParallelMath::ToUInt15(ParallelMath::Max(zeroS16, ParallelMath::LosslessCast<MSInt16>::Cast(unquantizedColor) - ParallelMath::MakeSInt16(modifier)));
            }

            MUInt16 signBits = ParallelMath::MakeUInt16(0);
            for (int px = 0; px < 16; px++)
            {
                MFloat errors[2];
                for (int i = 0; i < 2; i++)
                {
                    if (isFakeBT709)
                        errors[i] = ComputeErrorFakeBT709(colors[i], preWeightedPixels[px]);
                    else if (isUniform)
                        errors[i] = ComputeErrorUniform(colors[i], pixels[px]);
                    else
                        errors[i] = ComputeErrorWeighted(colors[i], preWeightedPixels[px], options);
                }

                ParallelMath::Int16CompFlag errorOneLess = ParallelMath::FloatFlagToInt16(ParallelMath::Less(errors[1], errors[0]));
                he.errors[ci][px] = ParallelMath::Min(errors[0], errors[1]);
                signBits = signBits | ParallelMath::SelectOrZero(errorOneLess, ParallelMath::MakeUInt16(1 << px));
            }
            he.signBits[ci] = signBits;
        }

        int maxUniqueColorCombos = 0;
        for (int block = 0; block < ParallelMath::ParallelSize; block++)
        {
            int numUniqueColorCombos = ParallelMath::Extract(he.numUniqueColors[0], block) * ParallelMath::Extract(he.numUniqueColors[1], block);
            if (numUniqueColorCombos > maxUniqueColorCombos)
                maxUniqueColorCombos = numUniqueColorCombos;
        }

        MUInt15 indexes[2] = { zero15, zero15 };
        MUInt15 maxIndex[2] = { he.numUniqueColors[0] - ParallelMath::MakeUInt15(1), he.numUniqueColors[1] - ParallelMath::MakeUInt15(1) };

        int block1Starts[ParallelMath::ParallelSize];
        for (int block = 0; block < ParallelMath::ParallelSize; block++)
            block1Starts[block] = ParallelMath::Extract(he.numUniqueColors[0], block);

        for (int combo = 0; combo < maxUniqueColorCombos; combo++)
        {
            MUInt15 index0 = indexes[0] + ParallelMath::MakeUInt15(1);
            ParallelMath::Int16CompFlag index0Overflow = ParallelMath::Less(maxIndex[0], index0);
            ParallelMath::ConditionalSet(index0, index0Overflow, ParallelMath::MakeUInt15(0));

            MUInt15 index1 = ParallelMath::Min(maxIndex[1], indexes[1] + ParallelMath::SelectOrZero(index0Overflow, ParallelMath::MakeUInt15(1)));
            indexes[0] = index0;
            indexes[1] = index1;

            int ci0[ParallelMath::ParallelSize];
            int ci1[ParallelMath::ParallelSize];
            MUInt15 color0;
            MUInt15 color1;

            for (int block = 0; block < ParallelMath::ParallelSize; block++)
            {
                ci0[block] = ParallelMath::Extract(index0, block);
                ci1[block] = ParallelMath::Extract(index1, block) + block1Starts[block];
                ParallelMath::PutUInt15(color0, block, ParallelMath::Extract(he.uniqueQuantizedColors[ci0[block]], block));
                ParallelMath::PutUInt15(color1, block, ParallelMath::Extract(he.uniqueQuantizedColors[ci1[block]], block));
            }

            MFloat totalError = ParallelMath::MakeFloatZero();
            MUInt16 sectorBits = ParallelMath::MakeUInt16(0);
            MUInt16 signBits = ParallelMath::MakeUInt16(0);
            for (int px = 0; px < 16; px++)
            {
                MFloat errorCI0;
                MFloat errorCI1;
                MUInt16 signBits0;
                MUInt16 signBits1;

                for (int block = 0; block < ParallelMath::ParallelSize; block++)
                {
                    ParallelMath::PutFloat(errorCI0, block, ParallelMath::Extract(he.errors[ci0[block]][px], block));
                    ParallelMath::PutFloat(errorCI1, block, ParallelMath::Extract(he.errors[ci1[block]][px], block));
                    ParallelMath::PutUInt16(signBits0, block, ParallelMath::Extract(he.signBits[ci0[block]], block));
                    ParallelMath::PutUInt16(signBits1, block, ParallelMath::Extract(he.signBits[ci1[block]], block));
                }

                totalError = totalError + ParallelMath::Min(errorCI0, errorCI1);

                MUInt16 bitPosition = ParallelMath::MakeUInt16(1 << px);

                ParallelMath::Int16CompFlag error1Better = ParallelMath::FloatFlagToInt16(ParallelMath::Less(errorCI1, errorCI0));

                sectorBits = sectorBits | ParallelMath::SelectOrZero(error1Better, bitPosition);
                signBits = signBits | (bitPosition & ParallelMath::Select(error1Better, signBits1, signBits0));
            }

            ParallelMath::FloatCompFlag totalErrorBetter = ParallelMath::Less(totalError, bestError);
            ParallelMath::Int16CompFlag totalErrorBetter16 = ParallelMath::FloatFlagToInt16(totalErrorBetter);
            if (ParallelMath::AnySet(totalErrorBetter16))
            {
                bestIsThisMode = bestIsThisMode | totalErrorBetter16;
                ParallelMath::ConditionalSet(bestTable, totalErrorBetter16, ParallelMath::MakeUInt15(table));
                ParallelMath::ConditionalSet(bestColors[0], totalErrorBetter16, color0);
                ParallelMath::ConditionalSet(bestColors[1], totalErrorBetter16, color1);
                ParallelMath::ConditionalSet(bestSectorBits, totalErrorBetter16, sectorBits);
                ParallelMath::ConditionalSet(bestSignBits, totalErrorBetter16, signBits);
                bestError = ParallelMath::Min(totalError, bestError);
            }
        }
    }

    const int selectorOrder[] = { 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15 };

    if (ParallelMath::AnySet(bestIsThisMode))
    {
        for (int block = 0; block < ParallelMath::ParallelSize; block++)
        {
            uint32_t lowBits = 0;
            uint32_t highBits = 0;

            ParallelMath::ScalarUInt16 blockBestColors[2] = { ParallelMath::Extract(bestColors[0], block), ParallelMath::Extract(bestColors[1], block) };

            if (blockBestColors[0] == blockBestColors[1])
                continue;	// TODO: Encode this as T mode instead

            int16_t colors[2][3];
            for (int sector = 0; sector < 2; sector++)
            {
                for (int ch = 0; ch < 3; ch++)
                    colors[sector][ch] = (blockBestColors[sector] >> ((2 - ch) * 5)) & 15;
            }

            ParallelMath::ScalarUInt16 blockBestSectorBits = ParallelMath::Extract(bestSectorBits, block);
            ParallelMath::ScalarUInt16 blockBestSignBits = ParallelMath::Extract(bestSignBits, block);
            ParallelMath::ScalarUInt16 blockBestTable = ParallelMath::Extract(bestTable, block);

            if (((blockBestTable & 1) == 1) != (blockBestColors[0] > blockBestColors[1]))
            {
                for (int ch = 0; ch < 3; ch++)
                    std::swap(colors[0][ch], colors[1][ch]);
                blockBestSectorBits ^= 0xffff;
            }

            int r1 = colors[0][0];
            int g1a = colors[0][1] >> 1;
            int g1b = (colors[0][1] & 1);
            int b1a = colors[0][2] >> 3;
            int b1b = colors[0][2] & 7;
            int r2 = colors[1][0];
            int g2 = colors[1][1];
            int b2 = colors[1][2];

            // Avoid overflowing R
            if ((g1a & 4) != 0 && r1 + g1a < 8)
                highBits |= 1 << (63 - 32);

            int fakeDG = b1b >> 1;
            int fakeG = b1a | (g1b << 1);

            if (fakeG + fakeDG < 4)
            {
                // Overflow low
                highBits |= 1 << (50 - 32);
            }
            else
            {
                // Overflow high
                highBits |= 7 << (53 - 32);
            }

            int da = (blockBestTable >> 2) & 1;
            int db = (blockBestTable >> 1) & 1;

            highBits |= r1 << (59 - 32);
            highBits |= g1a << (56 - 32);
            highBits |= g1b << (52 - 32);
            highBits |= b1a << (51 - 32);
            highBits |= b1b << (47 - 32);
            highBits |= r2 << (43 - 32);
            highBits |= g2 << (39 - 32);
            highBits |= b2 << (35 - 32);
            highBits |= da << (34 - 32);
            highBits |= 1 << (33 - 32);
            highBits |= db << (32 - 32);

            for (int px = 0; px < 16; px++)
            {
                int sectorBit = (blockBestSectorBits >> selectorOrder[px]) & 1;
                int signBit = (blockBestSignBits >> selectorOrder[px]) & 1;

                lowBits |= (signBit << px);
                lowBits |= (sectorBit << (16 + px));
            }

            uint8_t *output = outputBuffer + (block * 8);

            for (int i = 0; i < 4; i++)
                output[i] = (highBits >> (24 - i * 8)) & 0xff;
            for (int i = 0; i < 4; i++)
                output[i + 4] = (lowBits >> (24 - i * 8)) & 0xff;
        }
    }
}

cvtt::ParallelMath::UInt15 cvtt::Internal::ETCComputer::DecodePlanarCoeff(const MUInt15 &coeff, int ch)
{
    if (ch == 1)
        return (coeff << 1) | (ParallelMath::RightShift(coeff, 6));
    else
        return (coeff << 2) | (ParallelMath::RightShift(coeff, 4));
}

void cvtt::Internal::ETCComputer::EncodePlanar(uint8_t *outputBuffer, MFloat &bestError, const MUInt15 pixels[16][3], const MFloat preWeightedPixels[16][3], const Options &options)
{
    // NOTE: If it's desired to do this in another color space, the best way to do it would probably be
    // to do everything in that color space and then transform it back to RGB.

    // We compute H = (H-O)/4 and V= (V-O)/4 to simplify the math

    // error = (x*H + y*V + O - C)^2
    MFloat h[3] = { ParallelMath::MakeFloatZero(), ParallelMath::MakeFloatZero(), ParallelMath::MakeFloatZero() };
    MFloat v[3] = { ParallelMath::MakeFloatZero(), ParallelMath::MakeFloatZero(), ParallelMath::MakeFloatZero() };
    MFloat o[3] = { ParallelMath::MakeFloatZero(), ParallelMath::MakeFloatZero(), ParallelMath::MakeFloatZero() };

    bool isFakeBT709 = ((options.flags & cvtt::Flags::ETC_UseFakeBT709) != 0);
    bool isUniform = ((options.flags & cvtt::Flags::Uniform) != 0);

    MFloat totalError = ParallelMath::MakeFloatZero();
    MUInt15 bestCoeffs[3][3];	// [Channel][Coeff]
    for (int ch = 0; ch < 3; ch++)
    {
        float fhh = 0.f;
        float fho = 0.f;
        float fhv = 0.f;
        float foo = 0.f;
        float fov = 0.f;
        float fvv = 0.f;
        MFloat fc = ParallelMath::MakeFloatZero();
        MFloat fh = ParallelMath::MakeFloatZero();
        MFloat fv = ParallelMath::MakeFloatZero();
        MFloat fo = ParallelMath::MakeFloatZero();

        float &foh = fho;
        float &fvh = fhv;
        float &fvo = fov;

        for (int px = 0; px < 16; px++)
        {
            float x = static_cast<float>(px % 4);
            float y = static_cast<float>(px / 4);
            MFloat c = isFakeBT709 ? preWeightedPixels[px][ch] : ParallelMath::ToFloat(pixels[px][ch]);

            // (x*H + y*V + O - C)^2
            fhh += x * x;
            fhv += x * y;
            fho += x;
            fh = fh - c * x;

            fvh += y * x;
            fvv += y * y;
            fvo += y;
            fv = fv - c * y;

            foh += x;
            fov += y;
            foo += 1;
            fo = fo - c;

            fh = fh - c * x;
            fv = fv - c * y;
            fo = fo - c;
            fc = fc + c * c;
        }

        //float totalError = fhh * h * h + fho * h*o + fhv * h*v + foo * o * o + fov * o*v + fvv * v * v + fh * h + fv * v + fo * o + fc;

        // error = fhh*h^2 + fho*h*o + fhv*h*v + foo*o^2 + fov*o*v + fvv*v^2 + fh*h + fv*v + fo*o + fc
        // derror/dh = 2*fhh*h + fho*o + fhv*v + fh
        // derror/dv = fhv*h + fov*o + 2*fvv*v + fv
        // derror/do = fho*h + 2*foo*o + fov*v + fo

        // Solve system of equations
        // h o v 1 = 0
        // -------
        // d e f g  R0
        // i j k l  R1
        // m n p q  R2

        float d = 2.0f * fhh;
        float e = fho;
        float f = fhv;
        MFloat gD = fh;

        float i = fhv;
        float j = fov;
        float k = 2.0f * fvv;
        MFloat lD = fv;

        float m = fho;
        float n = 2.0f * foo;
        float p = fov;
        MFloat qD = fo;

        {
            // Factor out first column from R1 and R2
            float r0to1 = -i / d;
            float r0to2 = -m / d;

            // 0 j1 k1 l1D
            float j1 = j + r0to1 * e;
            float k1 = k + r0to1 * f;
            MFloat l1D = lD + gD * r0to1;

            // 0 n1 p1 q1D
            float n1 = n + r0to2 * e;
            float p1 = p + r0to2 * f;
            MFloat q1D = qD + gD * r0to2;

            // Factor out third column from R2
            float r1to2 = -p1 / k1;

            // 0 n2 0 q2D
            float n2 = n1 + r1to2 * j1;
            MFloat q2D = q1D + l1D * r1to2;

            o[ch] = -q2D / n2;

            // Factor out second column from R1
            // 0 n2 0 q2D

            float r2to1 = -j1 / n2;

            // 0 0 k1 l2D
            // 0 n2 0 q2D
            MFloat l2D = l1D + q2D * r2to1;

            float elim2 = -f / k1;
            float elim1 = -e / n2;

            // d 0 0 g2D
            MFloat g2D = gD + l2D * elim2 + q2D * elim1;

            // n2*o + q2 = 0
            // o = -q2 / n2
            h[ch] = -g2D / d;
            v[ch] = -l2D / k1;
        }

        // Undo the local transformation
        h[ch] = h[ch] * 4.0f + o[ch];
        v[ch] = v[ch] * 4.0f + o[ch];
    }

    if (isFakeBT709)
    {
        MFloat oRGB[3];
        MFloat hRGB[3];
        MFloat vRGB[3];

        ConvertFromFakeBT709(oRGB, o);
        ConvertFromFakeBT709(hRGB, h);
        ConvertFromFakeBT709(vRGB, v);

        // Twiddling in fake BT.607 is a mess, just round off for now (the precision is pretty good anyway)
        {
            ParallelMath::RoundTowardNearestForScope rtn;

            for (int ch = 0; ch < 3; ch++)
            {
                MFloat fcoeffs[3] = { oRGB[ch], hRGB[ch], vRGB[ch] };

                for (int c = 0; c < 3; c++)
                {
                    MFloat coeff = ParallelMath::Max(ParallelMath::MakeFloatZero(), fcoeffs[c]);
                    if (ch == 1)
                        coeff = ParallelMath::Min(ParallelMath::MakeFloat(127.0f), coeff * (127.0f / 255.0f));
                    else
                        coeff = ParallelMath::Min(ParallelMath::MakeFloat(63.0f), coeff * (63.0f / 255.0f));
                    fcoeffs[c] = coeff;
                }

                for (int c = 0; c < 3; c++)
                    bestCoeffs[ch][c] = ParallelMath::RoundAndConvertToU15(fcoeffs[c], &rtn);
            }
        }

        MUInt15 reconstructed[16][3];
        for (int ch = 0; ch < 3; ch++)
        {
            MUInt15 dO = DecodePlanarCoeff(bestCoeffs[ch][0], ch);
            MUInt15 dH = DecodePlanarCoeff(bestCoeffs[ch][1], ch);
            MUInt15 dV = DecodePlanarCoeff(bestCoeffs[ch][2], ch);

            MSInt16 hMinusO = ParallelMath::LosslessCast<MSInt16>::Cast(dH) - ParallelMath::LosslessCast<MSInt16>::Cast(dO);
            MSInt16 vMinusO = ParallelMath::LosslessCast<MSInt16>::Cast(dV) - ParallelMath::LosslessCast<MSInt16>::Cast(dO);

            MFloat error = ParallelMath::MakeFloatZero();

            MSInt16 addend = ParallelMath::LosslessCast<MSInt16>::Cast(dO << 2) + 2;

            for (int px = 0; px < 16; px++)
            {
                MUInt15 pxv = ParallelMath::MakeUInt15(px);
                MSInt16 x = ParallelMath::LosslessCast<MSInt16>::Cast(pxv & ParallelMath::MakeUInt15(3));
                MSInt16 y = ParallelMath::LosslessCast<MSInt16>::Cast(ParallelMath::RightShift(pxv, 2));

                MSInt16 interpolated = ParallelMath::RightShift(ParallelMath::CompactMultiply(x, hMinusO) + ParallelMath::CompactMultiply(y, vMinusO) + addend, 2);
                MUInt15 clampedLow = ParallelMath::ToUInt15(ParallelMath::Max(ParallelMath::MakeSInt16(0), interpolated));
                reconstructed[px][ch] = ParallelMath::Min(ParallelMath::MakeUInt15(255), clampedLow);
            }
        }

        totalError = ParallelMath::MakeFloatZero();
        for (int px = 0; px < 16; px++)
            totalError = totalError + ComputeErrorFakeBT709(reconstructed[px], preWeightedPixels[px]);
    }
    else
    {
        for (int ch = 0; ch < 3; ch++)
        {
            MFloat fcoeffs[3] = { o[ch], h[ch], v[ch] };
            MUInt15 coeffRanges[3][2];

            for (int c = 0; c < 3; c++)
            {
                MFloat coeff = ParallelMath::Max(ParallelMath::MakeFloatZero(), fcoeffs[c]);
                if (ch == 1)
                    coeff = ParallelMath::Min(ParallelMath::MakeFloat(127.0f), coeff * (127.0f / 255.0f));
                else
                    coeff = ParallelMath::Min(ParallelMath::MakeFloat(63.0f), coeff * (63.0f / 255.0f));
                fcoeffs[c] = coeff;
            }

            {
                ParallelMath::RoundDownForScope rd;
                for (int c = 0; c < 3; c++)
                    coeffRanges[c][0] = ParallelMath::RoundAndConvertToU15(fcoeffs[c], &rd);
            }

            {
                ParallelMath::RoundUpForScope ru;
                for (int c = 0; c < 3; c++)
                    coeffRanges[c][1] = ParallelMath::RoundAndConvertToU15(fcoeffs[c], &ru);
            }

            MFloat bestChannelError = ParallelMath::MakeFloat(FLT_MAX);
            for (int io = 0; io < 2; io++)
            {
                MUInt15 dO = DecodePlanarCoeff(coeffRanges[0][io], ch);

                for (int ih = 0; ih < 2; ih++)
                {
                    MUInt15 dH = DecodePlanarCoeff(coeffRanges[1][ih], ch);
                    MSInt16 hMinusO = ParallelMath::LosslessCast<MSInt16>::Cast(dH) - ParallelMath::LosslessCast<MSInt16>::Cast(dO);

                    for (int iv = 0; iv < 2; iv++)
                    {
                        MUInt15 dV = DecodePlanarCoeff(coeffRanges[2][iv], ch);
                        MSInt16 vMinusO = ParallelMath::LosslessCast<MSInt16>::Cast(dV) - ParallelMath::LosslessCast<MSInt16>::Cast(dO);

                        MFloat error = ParallelMath::MakeFloatZero();

                        MSInt16 addend = ParallelMath::LosslessCast<MSInt16>::Cast(dO << 2) + 2;

                        for (int px = 0; px < 16; px++)
                        {
                            MUInt15 pxv = ParallelMath::MakeUInt15(px);
                            MSInt16 x = ParallelMath::LosslessCast<MSInt16>::Cast(pxv & ParallelMath::MakeUInt15(3));
                            MSInt16 y = ParallelMath::LosslessCast<MSInt16>::Cast(ParallelMath::RightShift(pxv, 2));

                            MSInt16 interpolated = ParallelMath::RightShift(ParallelMath::CompactMultiply(x, hMinusO) + ParallelMath::CompactMultiply(y, vMinusO) + addend, 2);
                            MUInt15 clampedLow = ParallelMath::ToUInt15(ParallelMath::Max(ParallelMath::MakeSInt16(0), interpolated));
                            MUInt15 dec = ParallelMath::Min(ParallelMath::MakeUInt15(255), clampedLow);

                            MSInt16 delta = ParallelMath::LosslessCast<MSInt16>::Cast(pixels[px][ch]) - ParallelMath::LosslessCast<MSInt16>::Cast(dec);

                            MFloat deltaF = ParallelMath::ToFloat(delta);
                            error = error + deltaF * deltaF;
                        }

                        ParallelMath::Int16CompFlag errorBetter = ParallelMath::FloatFlagToInt16(ParallelMath::Less(error, bestChannelError));
                        if (ParallelMath::AnySet(errorBetter))
                        {
                            bestChannelError = ParallelMath::Min(error, bestChannelError);
                            ParallelMath::ConditionalSet(bestCoeffs[ch][0], errorBetter, coeffRanges[0][io]);
                            ParallelMath::ConditionalSet(bestCoeffs[ch][1], errorBetter, coeffRanges[1][ih]);
                            ParallelMath::ConditionalSet(bestCoeffs[ch][2], errorBetter, coeffRanges[2][iv]);
                        }
                    }
                }
            }

            if (!isUniform)
            {
                switch (ch)
                {
                case 0:
                    bestChannelError = bestChannelError * (options.redWeight * options.redWeight);
                    break;
                case 1:
                    bestChannelError = bestChannelError * (options.greenWeight * options.greenWeight);
                    break;
                case 2:
                    bestChannelError = bestChannelError * (options.blueWeight * options.blueWeight);
                    break;
                default:
                    break;
                }
            }

            totalError = totalError + bestChannelError;
        }
    }

    ParallelMath::Int16CompFlag errorBetter = ParallelMath::FloatFlagToInt16(ParallelMath::Less(totalError, bestError));
    if (ParallelMath::AnySet(errorBetter))
    {
        bestError = ParallelMath::Min(bestError, totalError);

        for (int block = 0; block < ParallelMath::ParallelSize; block++)
        {
            if (!ParallelMath::Extract(errorBetter, block))
                continue;

            int ro = ParallelMath::Extract(bestCoeffs[0][0], block);
            int rh = ParallelMath::Extract(bestCoeffs[0][1], block);
            int rv = ParallelMath::Extract(bestCoeffs[0][2], block);

            int go = ParallelMath::Extract(bestCoeffs[1][0], block);
            int gh = ParallelMath::Extract(bestCoeffs[1][1], block);
            int gv = ParallelMath::Extract(bestCoeffs[1][2], block);

            int bo = ParallelMath::Extract(bestCoeffs[2][0], block);
            int bh = ParallelMath::Extract(bestCoeffs[2][1], block);
            int bv = ParallelMath::Extract(bestCoeffs[2][2], block);

            int go1 = go >> 6;
            int go2 = go & 63;

            int bo1 = bo >> 5;
            int bo2 = (bo >> 3) & 3;
            int bo3 = bo & 7;

            int rh1 = (rh >> 1);
            int rh2 = rh & 1;

            int fakeR = ro >> 2;
            int fakeDR = go1 | ((ro & 3) << 1);

            int fakeG = (go2 >> 2);
            int fakeDG = ((go2 & 3) << 1) | bo1;

            int fakeB = bo2;
            int fakeDB = bo3 >> 1;

            uint32_t highBits = 0;
            uint32_t lowBits = 0;

            // Avoid overflowing R
            if ((fakeDR & 4) != 0 && fakeR + fakeDR < 8)
                highBits |= 1 << (63 - 32);

            // Avoid overflowing G
            if ((fakeDG & 4) != 0 && fakeG + fakeDG < 8)
                highBits |= 1 << (55 - 32);

            // Overflow B
            if (fakeB + fakeDB < 4)
            {
                // Overflow low
                highBits |= 1 << (42 - 32);
            }
            else
            {
                // Overflow high
                highBits |= 7 << (45 - 32);
            }

            highBits |= ro << (57 - 32);
            highBits |= go1 << (56 - 32);
            highBits |= go2 << (49 - 32);
            highBits |= bo1 << (48 - 32);
            highBits |= bo2 << (43 - 32);
            highBits |= bo3 << (39 - 32);
            highBits |= rh1 << (34 - 32);
            highBits |= 1 << (33 - 32);
            highBits |= rh2 << (32 - 32);

            lowBits |= gh << 25;
            lowBits |= bh << 19;
            lowBits |= rv << 13;
            lowBits |= gv << 6;
            lowBits |= bv << 0;

            for (int i = 0; i < 4; i++)
                outputBuffer[block * 8 + i] = (highBits >> (24 - i * 8)) & 0xff;
            for (int i = 0; i < 4; i++)
                outputBuffer[block * 8 + i + 4] = (lowBits >> (24 - i * 8)) & 0xff;
        }
    }
}

void cvtt::Internal::ETCComputer::CompressETC2Block(uint8_t *outputBuffer, const PixelBlockU8 *pixelBlocks, ETC2CompressionData *compressionData, const Options &options)
{
    MFloat bestError = ParallelMath::MakeFloat(FLT_MAX);

    ETC2CompressionDataInternal* internalData = static_cast<ETC2CompressionDataInternal*>(compressionData);

    MUInt15 pixels[16][3];
    MFloat preWeightedPixels[16][3];
    ExtractBlocks(pixels, preWeightedPixels, pixelBlocks, options);

    EncodePlanar(outputBuffer, bestError, pixels, preWeightedPixels, options);

    MSInt16 chromaCoordinates3[16][2];
    for (int px = 0; px < 16; px++)
    {
        chromaCoordinates3[px][0] = ParallelMath::LosslessCast<MSInt16>::Cast(pixels[px][0]) - ParallelMath::LosslessCast<MSInt16>::Cast(pixels[px][2]);
        chromaCoordinates3[px][1] = ParallelMath::LosslessCast<MSInt16>::Cast(pixels[px][0]) - ParallelMath::LosslessCast<MSInt16>::Cast(pixels[px][1] << 1) + ParallelMath::LosslessCast<MSInt16>::Cast(pixels[px][2]);
    }

    MSInt16 chromaCoordinateCentroid[2] = { ParallelMath::MakeSInt16(0), ParallelMath::MakeSInt16(0) };
    for (int px = 0; px < 16; px++)
    {
        for (int ch = 0; ch < 2; ch++)
            chromaCoordinateCentroid[ch] = chromaCoordinateCentroid[ch] + chromaCoordinates3[px][ch];
    }

    MSInt16 chromaDelta[16][2];
    for (int px = 0; px < 16; px++)
    {
        for (int ch = 0; ch < 2; ch++)
            chromaDelta[px][ch] = (chromaCoordinates3[px][ch] << 4) - chromaCoordinateCentroid[ch];
    }

    const MFloat rcpSqrt3 = ParallelMath::MakeFloat(0.57735026918962576450914878050196f);

    MFloat covXX = ParallelMath::MakeFloatZero();
    MFloat covYY = ParallelMath::MakeFloatZero();
    MFloat covXY = ParallelMath::MakeFloatZero();

    for (int px = 0; px < 16; px++)
    {
        MFloat nx = ParallelMath::ToFloat(chromaDelta[px][0]);
        MFloat ny = ParallelMath::ToFloat(chromaDelta[px][1]) * rcpSqrt3;

        covXX = covXX + nx * nx;
        covYY = covYY + ny * ny;
        covXY = covXY + nx * ny;
    }

    MFloat halfTrace = (covXX + covYY) * 0.5f;
    MFloat det = covXX * covYY - covXY * covXY;

    MFloat mm = ParallelMath::Sqrt(ParallelMath::Max(ParallelMath::MakeFloatZero(), halfTrace * halfTrace - det));

    MFloat ev = halfTrace + mm;

    MFloat dx = (covYY - ev + covXY);
    MFloat dy = -(covXX - ev + covXY);

    // If evenly distributed, pick an arbitrary plane
    ParallelMath::FloatCompFlag allZero = ParallelMath::Equal(dx, ParallelMath::MakeFloatZero()) & ParallelMath::Equal(dy, ParallelMath::MakeFloatZero());
    ParallelMath::ConditionalSet(dx, allZero, ParallelMath::MakeFloat(1.f));

    ParallelMath::Int16CompFlag sectorAssignments[16];
    for (int px = 0; px < 16; px++)
        sectorAssignments[px] = ParallelMath::FloatFlagToInt16(ParallelMath::Less(ParallelMath::ToFloat(chromaDelta[px][0]) * dx + ParallelMath::ToFloat(chromaDelta[px][1]) * dy * rcpSqrt3, ParallelMath::MakeFloatZero()));

    EncodeTMode(outputBuffer, bestError, sectorAssignments, pixels, preWeightedPixels, options);

    // Flip sector assignments
    for (int px = 0; px < 16; px++)
        sectorAssignments[px] = ParallelMath::Not(sectorAssignments[px]);

    EncodeTMode(outputBuffer, bestError, sectorAssignments, pixels, preWeightedPixels, options);

    EncodeHMode(outputBuffer, bestError, sectorAssignments, pixels, internalData->m_h, preWeightedPixels, options);

    CompressETC1BlockInternal(bestError, outputBuffer, pixels, preWeightedPixels, internalData->m_drs, options);
}

void cvtt::Internal::ETCComputer::CompressETC2AlphaBlock(uint8_t *outputBuffer, const PixelBlockU8 *pixelBlocks, const Options &options)
{
    MUInt15 pixels[16];
    MUInt15 minAlpha = ParallelMath::MakeUInt15(255);
    MUInt15 maxAlpha = ParallelMath::MakeUInt15(0);

    for (int px = 0; px < 16; px++)
    {
        for (int block = 0; block < ParallelMath::ParallelSize; block++)
            ParallelMath::PutUInt15(pixels[px], block, pixelBlocks[block].m_pixels[px][3]);
    }

    for (int px = 0; px < 16; px++)
    {
        minAlpha = ParallelMath::Min(minAlpha, pixels[px]);
        maxAlpha = ParallelMath::Max(maxAlpha, pixels[px]);
    }

    MUInt15 alphaSpan = maxAlpha - minAlpha;
    MUInt15 alphaSpanMidpointTimes2 = maxAlpha + minAlpha;

    MUInt31 bestTotalError = ParallelMath::MakeUInt31(0x7fffffff);
    MUInt15 bestTableIndex = ParallelMath::MakeUInt15(0);
    MUInt15 bestBaseCodeword = ParallelMath::MakeUInt15(0);
    MUInt15 bestMultiplier = ParallelMath::MakeUInt15(0);
    MUInt15 bestIndexes[16];

    for (int px = 0; px < 16; px++)
        bestIndexes[px] = ParallelMath::MakeUInt15(0);

    const int numAlphaRanges = 10;
    for (uint16_t tableIndex = 0; tableIndex < 16; tableIndex++)
    {
        for (int r = 0; r < numAlphaRanges; r++)
        {
            int subrange = r % 3;
            int mainRange = r / 3;

            int16_t maxOffset = Tables::ETC2::g_alphaModifierTablePositive[tableIndex][3 - mainRange - (subrange & 1)];
            int16_t minOffset = -Tables::ETC2::g_alphaModifierTablePositive[tableIndex][3 - mainRange - ((subrange >> 1) & 1)] - 1;
            uint16_t offsetSpan = static_cast<uint16_t>(maxOffset - minOffset);

            MSInt16 vminOffset = ParallelMath::MakeSInt16(minOffset);
            MUInt15 vmaxOffset = ParallelMath::MakeUInt15(maxOffset);
            MUInt15 voffsetSpan = ParallelMath::MakeUInt15(offsetSpan);

            MUInt15 minMultiplier = ParallelMath::MakeUInt15(0);
            for (int block = 0; block < ParallelMath::ParallelSize; block++)
            {
                uint16_t singleAlphaSpan = ParallelMath::Extract(alphaSpan, block);

                uint16_t lowMultiplier = singleAlphaSpan / offsetSpan;
                ParallelMath::PutUInt15(minMultiplier, block, lowMultiplier);
            }

            // We cap at 1 and 14 so both multipliers are valid and dividable
            // Cases where offset span is 0 should be caught by multiplier 1 of table 13
            minMultiplier = ParallelMath::Max(ParallelMath::Min(minMultiplier, ParallelMath::MakeUInt15(14)), ParallelMath::MakeUInt15(1));

            for (uint16_t multiplierOffset = 0; multiplierOffset < 2; multiplierOffset++)
            {
                MUInt15 multiplier = ParallelMath::MakeUInt15(multiplierOffset) + minMultiplier;

                MSInt16 multipliedMinOffset = ParallelMath::CompactMultiply(ParallelMath::LosslessCast<MSInt16>::Cast(multiplier), vminOffset);
                MUInt15 multipliedMaxOffset = ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::CompactMultiply(multiplier, vmaxOffset));

                // codeword = (maxOffset + minOffset + minAlpha + maxAlpha) / 2
                MSInt16 unclampedBaseAlphaTimes2 = ParallelMath::LosslessCast<MSInt16>::Cast(alphaSpanMidpointTimes2) - ParallelMath::LosslessCast<MSInt16>::Cast(multipliedMaxOffset) - multipliedMinOffset;

                MUInt15 clampedBaseAlphaTimes2 = ParallelMath::Min(ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::Max(unclampedBaseAlphaTimes2, ParallelMath::MakeSInt16(0))), ParallelMath::MakeUInt15(510));
                MUInt15 baseAlpha = ParallelMath::RightShift(clampedBaseAlphaTimes2 + ParallelMath::MakeUInt15(1), 1);

                MUInt15 indexes[16];
                MUInt31 totalError = ParallelMath::MakeUInt31(0);
                for (int px = 0; px < 16; px++)
                {
                    MUInt15 quantizedValues;
                    QuantizeETC2Alpha(tableIndex, pixels[px], baseAlpha, multiplier, indexes[px], quantizedValues);

                    totalError = totalError + ParallelMath::ToUInt31(ParallelMath::SqDiffUInt8(quantizedValues, pixels[px]));
                }

                ParallelMath::Int16CompFlag isBetter = ParallelMath::Int32FlagToInt16(ParallelMath::Less(totalError, bestTotalError));
                if (ParallelMath::AnySet(isBetter))
                {
                    ParallelMath::ConditionalSet(bestTotalError, isBetter, totalError);
                    ParallelMath::ConditionalSet(bestTableIndex, isBetter, ParallelMath::MakeUInt15(tableIndex));
                    ParallelMath::ConditionalSet(bestBaseCodeword, isBetter, baseAlpha);
                    ParallelMath::ConditionalSet(bestMultiplier, isBetter, multiplier);

                    for (int px = 0; px < 16; px++)
                        ParallelMath::ConditionalSet(bestIndexes[px], isBetter, indexes[px]);
                }

                // TODO: Do one refine pass
            }
        }
    }

    for (int block = 0; block < ParallelMath::ParallelSize; block++)
    {
        uint8_t *output = outputBuffer + block * 8;
        output[0] = static_cast<uint8_t>(ParallelMath::Extract(bestBaseCodeword, block));

        ParallelMath::ScalarUInt16 multiplier = ParallelMath::Extract(bestMultiplier, block);
        ParallelMath::ScalarUInt16 tableIndex = ParallelMath::Extract(bestTableIndex, block);

        output[1] = static_cast<uint8_t>((multiplier << 4) | tableIndex);

        static const int pixelSelectorOrder[16] = { 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15 };

        ParallelMath::ScalarUInt16 indexes[16];
        for (int px = 0; px < 16; px++)
            indexes[pixelSelectorOrder[px]] = ParallelMath::Extract(bestIndexes[px], block);

        int outputOffset = 2;
        int outputBits = 0;
        int numOutputBits = 0;
        for (int s = 0; s < 16; s++)
        {
            outputBits = (outputBits << 3) | indexes[s];
            numOutputBits += 3;

            if (numOutputBits >= 8)
            {
                output[outputOffset++] = static_cast<uint8_t>(outputBits >> (numOutputBits - 8));
                numOutputBits -= 8;

                outputBits &= ((1 << numOutputBits) - 1);
            }
        }

        assert(outputOffset == 8 && numOutputBits == 0);
    }
}

void cvtt::Internal::ETCComputer::CompressETC1Block(uint8_t *outputBuffer, const PixelBlockU8 *inputBlocks, ETC1CompressionData *compressionData, const Options &options)
{
    DifferentialResolveStorage &drs = static_cast<ETC1CompressionDataInternal*>(compressionData)->m_drs;
    MFloat bestTotalError = ParallelMath::MakeFloat(FLT_MAX);

    MUInt15 pixels[16][3];
    MFloat preWeightedPixels[16][3];
    ExtractBlocks(pixels, preWeightedPixels, inputBlocks, options);

    CompressETC1BlockInternal(bestTotalError, outputBuffer, pixels, preWeightedPixels, drs, options);
}

void cvtt::Internal::ETCComputer::ExtractBlocks(MUInt15 pixels[16][3], MFloat preWeightedPixels[16][3], const PixelBlockU8 *inputBlocks, const Options &options)
{
    bool isFakeBT709 = ((options.flags & cvtt::Flags::ETC_UseFakeBT709) != 0);
    bool isUniform = ((options.flags & cvtt::Flags::Uniform) != 0);

    for (int px = 0; px < 16; px++)
    {
        for (int ch = 0; ch < 3; ch++)
        {
            for (int block = 0; block < ParallelMath::ParallelSize; block++)
                ParallelMath::PutUInt15(pixels[px][ch], block, inputBlocks[block].m_pixels[px][ch]);
        }

        if (isFakeBT709)
            ConvertToFakeBT709(preWeightedPixels[px], pixels[px]);
        else if (isUniform)
        {
            for (int ch = 0; ch < 3; ch++)
                preWeightedPixels[px][ch] = ParallelMath::ToFloat(pixels[px][ch]);
        }
        else
        {
            preWeightedPixels[px][0] = ParallelMath::ToFloat(pixels[px][0]) * options.redWeight;
            preWeightedPixels[px][1] = ParallelMath::ToFloat(pixels[px][1]) * options.greenWeight;
            preWeightedPixels[px][2] = ParallelMath::ToFloat(pixels[px][2]) * options.blueWeight;
        }
    }
}

void cvtt::Internal::ETCComputer::ResolveHalfBlockFakeBT709RoundingAccurate(MUInt15 quantized[3], const MUInt15 sectorCumulative[3], bool isDifferential)
{
    for (int ch = 0; ch < 3; ch++)
    {
        const MUInt15& cu15 = sectorCumulative[ch];

        if (isDifferential)
        {
            //quantized[ch] = (cu * 31 + (cu >> 3)) >> 11;
            quantized[ch] = ParallelMath::ToUInt15(
                ParallelMath::RightShift(
                (ParallelMath::LosslessCast<MUInt16>::Cast(cu15) << 5) - ParallelMath::LosslessCast<MUInt16>::Cast(cu15) + ParallelMath::LosslessCast<MUInt16>::Cast(ParallelMath::RightShift(cu15, 3))
                    , 11)
            );
        }
        else
        {
            //quantized[ch] = (cu * 30 + (cu >> 3)) >> 12;
            quantized[ch] = ParallelMath::ToUInt15(
                ParallelMath::RightShift(
                (ParallelMath::LosslessCast<MUInt16>::Cast(cu15) << 5) - ParallelMath::LosslessCast<MUInt16>::Cast(cu15 << 1) + ParallelMath::LosslessCast<MUInt16>::Cast(ParallelMath::RightShift(cu15, 3))
                    , 12)
            );
        }
    }

    MFloat lowOctantRGBFloat[3];
    MFloat highOctantRGBFloat[3];

    for (int ch = 0; ch < 3; ch++)
    {
        MUInt15 unquantized;
        MUInt15 unquantizedNext;
        if (isDifferential)
        {
            unquantized = (quantized[ch] << 3) | ParallelMath::RightShift(quantized[ch], 2);
            MUInt15 quantizedNext = ParallelMath::Min(ParallelMath::MakeUInt15(31), quantized[ch] + ParallelMath::MakeUInt15(1));
            unquantizedNext = (quantizedNext << 3) | ParallelMath::RightShift(quantizedNext, 2);
        }
        else
        {
            unquantized = (quantized[ch] << 4) | quantized[ch];
            unquantizedNext = ParallelMath::Min(ParallelMath::MakeUInt15(255), unquantized + ParallelMath::MakeUInt15(17));
        }
        lowOctantRGBFloat[ch] = ParallelMath::ToFloat(unquantized << 3);
        highOctantRGBFloat[ch] = ParallelMath::ToFloat(unquantizedNext << 3);
    }

    MFloat bestError = ParallelMath::MakeFloat(FLT_MAX);
    MUInt15 bestOctant = ParallelMath::MakeUInt15(0);

    MFloat cumulativeYUV[3];
    ConvertToFakeBT709(cumulativeYUV, sectorCumulative);

    for (uint16_t octant = 0; octant < 8; octant++)
    {
        const MFloat &r = (octant & 1) ? highOctantRGBFloat[0] : lowOctantRGBFloat[0];
        const MFloat &g = (octant & 2) ? highOctantRGBFloat[1] : lowOctantRGBFloat[1];
        const MFloat &b = (octant & 4) ? highOctantRGBFloat[2] : lowOctantRGBFloat[2];

        MFloat octantYUV[3];
        ConvertToFakeBT709(octantYUV, r, g, b);

        MFloat delta[3];
        for (int ch = 0; ch < 3; ch++)
            delta[ch] = octantYUV[ch] - cumulativeYUV[ch];

        MFloat error = delta[0] * delta[0] + delta[1] + delta[1] + delta[2] * delta[2];
        ParallelMath::Int16CompFlag errorBetter = ParallelMath::FloatFlagToInt16(ParallelMath::Less(error, bestError));
        ParallelMath::ConditionalSet(bestOctant, errorBetter, ParallelMath::MakeUInt15(octant));
        bestError = ParallelMath::Min(error, bestError);
    }

    for (int ch = 0; ch < 3; ch++)
        quantized[ch] = quantized[ch] + (ParallelMath::RightShift(bestOctant, ch) & ParallelMath::MakeUInt15(1));
}

void cvtt::Internal::ETCComputer::ResolveHalfBlockFakeBT709RoundingFast(MUInt15 quantized[3], const MUInt15 sectorCumulative[3], bool isDifferential)
{
    // sectorCumulative range is 0..2040 (11 bits)
    MUInt15 roundingOffset = ParallelMath::MakeUInt15(0);

    MUInt15 rOffset;
    MUInt15 gOffset;
    MUInt15 bOffset;
    MUInt15 quantizedBase[3];
    MUInt15 upperBound;

    if (isDifferential)
    {
        MUInt15 rOffset = (sectorCumulative[0] << 6);
        MUInt15 gOffset = (sectorCumulative[1] << 2);
        MUInt15 bOffset = ParallelMath::RightShift(sectorCumulative[2], 2);

        for (int ch = 0; ch < 3; ch++)
            quantizedBase[ch] = ParallelMath::RightShift(sectorCumulative[ch], 6);

        upperBound = ParallelMath::MakeUInt15(31);
    }
    else
    {
        MUInt15 rOffset = (sectorCumulative[0] << 5);
        MUInt15 gOffset = (sectorCumulative[1] << 1);
        MUInt15 bOffset = ParallelMath::RightShift(sectorCumulative[2], 3);

        for (int ch = 0; ch < 3; ch++)
            quantizedBase[ch] = ParallelMath::RightShift(sectorCumulative[ch], 7);

        upperBound = ParallelMath::MakeUInt15(15);
    }

    MUInt15 lookupIndex = (rOffset | gOffset | bOffset);

    MUInt15 octant;
    for (int block = 0; block < ParallelMath::ParallelSize; block++)
        ParallelMath::PutUInt15(octant, block, Tables::FakeBT709::g_rounding16[ParallelMath::Extract(lookupIndex, block)]);

    quantizedBase[0] = quantizedBase[0] + (octant & ParallelMath::MakeUInt15(1));
    quantizedBase[1] = quantizedBase[1] + (ParallelMath::RightShift(octant, 1) & ParallelMath::MakeUInt15(1));
    quantizedBase[2] = quantizedBase[2] + (ParallelMath::RightShift(octant, 2) & ParallelMath::MakeUInt15(1));

    for (int ch = 0; ch < 3; ch++)
        quantized[ch] = ParallelMath::Min(quantizedBase[ch], upperBound);
}

void cvtt::Internal::ETCComputer::ResolveTHFakeBT709Rounding(MUInt15 quantized[3], const MUInt15 targets[3], const MUInt15 &granularity)
{
    MFloat lowOctantRGBFloat[3];
    MFloat highOctantRGBFloat[3];

    for (int ch = 0; ch < 3; ch++)
    {
        MUInt15 unquantized = (quantized[ch] << 4) | quantized[ch];
        MUInt15 unquantizedNext = ParallelMath::Min(ParallelMath::MakeUInt15(255), unquantized + ParallelMath::MakeUInt15(17));

        lowOctantRGBFloat[ch] = ParallelMath::ToFloat(ParallelMath::CompactMultiply(unquantized, granularity) << 1);
        highOctantRGBFloat[ch] = ParallelMath::ToFloat(ParallelMath::CompactMultiply(unquantizedNext, granularity) << 1);
    }

    MFloat bestError = ParallelMath::MakeFloat(FLT_MAX);
    MUInt15 bestOctant = ParallelMath::MakeUInt15(0);

    MFloat cumulativeYUV[3];
    ConvertToFakeBT709(cumulativeYUV, ParallelMath::ToFloat(targets[0]), ParallelMath::ToFloat(targets[1]), ParallelMath::ToFloat(targets[2]));

    for (uint16_t octant = 0; octant < 8; octant++)
    {
        const MFloat &r = (octant & 1) ? highOctantRGBFloat[0] : lowOctantRGBFloat[0];
        const MFloat &g = (octant & 2) ? highOctantRGBFloat[1] : lowOctantRGBFloat[1];
        const MFloat &b = (octant & 4) ? highOctantRGBFloat[2] : lowOctantRGBFloat[2];

        MFloat octantYUV[3];
        ConvertToFakeBT709(octantYUV, r, g, b);

        MFloat delta[3];
        for (int ch = 0; ch < 3; ch++)
            delta[ch] = octantYUV[ch] - cumulativeYUV[ch];

        MFloat error = delta[0] * delta[0] + delta[1] + delta[1] + delta[2] * delta[2];
        ParallelMath::Int16CompFlag errorBetter = ParallelMath::FloatFlagToInt16(ParallelMath::Less(error, bestError));
        ParallelMath::ConditionalSet(bestOctant, errorBetter, ParallelMath::MakeUInt15(octant));
        bestError = ParallelMath::Min(error, bestError);
    }

    for (int ch = 0; ch < 3; ch++)
        quantized[ch] = quantized[ch] + (ParallelMath::RightShift(bestOctant, ch) & ParallelMath::MakeUInt15(1));
}

void cvtt::Internal::ETCComputer::ConvertToFakeBT709(MFloat yuv[3], const MUInt15 color[3])
{
    MFloat floatRGB[3];
    for (int ch = 0; ch < 3; ch++)
        floatRGB[ch] = ParallelMath::ToFloat(color[ch]);

    ConvertToFakeBT709(yuv, floatRGB);
}

void cvtt::Internal::ETCComputer::ConvertToFakeBT709(MFloat yuv[3], const MFloat color[3])
{
    ConvertToFakeBT709(yuv, color[0], color[1], color[2]);
}

void cvtt::Internal::ETCComputer::ConvertToFakeBT709(MFloat yuv[3], const MFloat &pr, const MFloat &pg, const MFloat &pb)
{
    MFloat r = pr;
    MFloat g = pg;
    MFloat b = pb;

    yuv[0] = r * 0.368233989135369f + g * 1.23876274963149f + b * 0.125054068802017f;
    yuv[1] = r * 0.5f - g * 0.4541529f - b * 0.04584709f;
    yuv[2] = r * -0.081014709086133f - g * 0.272538676238785f + b * 0.353553390593274f;
}

void cvtt::Internal::ETCComputer::ConvertFromFakeBT709(MFloat rgb[3], const MFloat yuv[3])
{
    MFloat yy = yuv[0] * 0.57735026466774571071f;
    MFloat u = yuv[1];
    MFloat v = yuv[2];

    rgb[0] = yy + u * 1.5748000207960953486f;
    rgb[1] = yy - u * 0.46812425854364753669f - v * 0.26491652528157560861f;
    rgb[2] = yy + v * 2.6242146882856944069f;
}


void cvtt::Internal::ETCComputer::QuantizeETC2Alpha(int tableIndex, const MUInt15& value, const MUInt15& baseValue, const MUInt15& multiplier, MUInt15& outIndexes, MUInt15& outQuantizedValues)
{
    MSInt16 offset = ParallelMath::LosslessCast<MSInt16>::Cast(value) - ParallelMath::LosslessCast<MSInt16>::Cast(baseValue);
    MSInt16 offsetTimes2 = offset + offset;

    // ETC2's offset tables all have a reflect about 0.5*multiplier
    MSInt16 offsetAboutReflectorTimes2 = offsetTimes2 + ParallelMath::LosslessCast<MSInt16>::Cast(multiplier);

    MUInt15 absOffsetAboutReflectorTimes2 = ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::Abs(offsetAboutReflectorTimes2));
    MUInt15 lookupIndex = ParallelMath::RightShift(absOffsetAboutReflectorTimes2, 1);

    MUInt15 positiveIndex;
    MUInt15 positiveOffsetUnmultiplied;
    for (int block = 0; block < ParallelMath::ParallelSize; block++)
    {
        uint16_t blockLookupIndex = ParallelMath::Extract(lookupIndex, block) / ParallelMath::Extract(multiplier, block);
        if (blockLookupIndex >= Tables::ETC2::g_alphaRoundingTableWidth)
            blockLookupIndex = Tables::ETC2::g_alphaRoundingTableWidth - 1;
        uint16_t index = Tables::ETC2::g_alphaRoundingTables[tableIndex][blockLookupIndex];
        ParallelMath::PutUInt15(positiveIndex, block, index);
        ParallelMath::PutUInt15(positiveOffsetUnmultiplied, block, Tables::ETC2::g_alphaModifierTablePositive[tableIndex][index]);

        // TODO: This is suboptimal when the offset is capped.  We should detect 0 and 255 values and always map them to the maximum offsets.
        // Doing that will also affect refinement though.
    }

    MSInt16 signBits = ParallelMath::RightShift(offsetAboutReflectorTimes2, 15);
    MSInt16 offsetUnmultiplied = ParallelMath::LosslessCast<MSInt16>::Cast(positiveOffsetUnmultiplied) ^ signBits;
    MSInt16 quantizedOffset = ParallelMath::CompactMultiply(offsetUnmultiplied, multiplier);

    MSInt16 offsetValue = ParallelMath::LosslessCast<MSInt16>::Cast(baseValue) + quantizedOffset;

    outQuantizedValues = ParallelMath::Min(ParallelMath::MakeUInt15(255), ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::Max(ParallelMath::MakeSInt16(0), offsetValue)));

    MUInt15 indexSub = ParallelMath::LosslessCast<MUInt15>::Cast(signBits) & ParallelMath::MakeUInt15(4);

    outIndexes = positiveIndex + ParallelMath::MakeUInt15(4) - indexSub;
}

void cvtt::Internal::ETCComputer::CompressETC1BlockInternal(MFloat &bestTotalError, uint8_t *outputBuffer, const MUInt15 pixels[16][3], const MFloat preWeightedPixels[16][3], DifferentialResolveStorage &drs, const Options &options)
{
	const int flipTables[2][2][8] =
	{
		{
			{ 0, 1, 4, 5, 8, 9, 12, 13 },
			{ 2, 3, 6, 7, 10, 11, 14, 15 }
		},
		{
			{ 0, 1, 2, 3, 4, 5, 6, 7 },
			{ 8, 9, 10, 11, 12, 13, 14, 15 }
		},
	};

	int numTries = 0;

    MUInt15 zeroU15 = ParallelMath::MakeUInt15(0);
    MUInt16 zeroU16 = ParallelMath::MakeUInt16(0);

    MUInt15 bestColors[2] = { zeroU15, zeroU15 };
    MUInt16 bestSelectors[2] = { zeroU16, zeroU16 };
    MUInt15 bestTables[2] = { zeroU15, zeroU15 };
    MUInt15 bestFlip = zeroU15;
    MUInt15 bestD = zeroU15;

    MUInt15 sectorPixels[2][2][8][3];
    MFloat sectorPreWeightedPixels[2][2][8][3];
    MUInt15 sectorCumulative[2][2][3];

    ParallelMath::Int16CompFlag bestIsThisMode = ParallelMath::MakeBoolInt16(false);

    for (int flip = 0; flip < 2; flip++)
	{
		for (int sector = 0; sector < 2; sector++)
		{
			for (int ch = 0; ch < 3; ch++)
				sectorCumulative[flip][sector][ch] = zeroU15;

			for (int px = 0; px < 8; px++)
			{
				for (int ch = 0; ch < 3; ch++)
				{
					MUInt15 pixelChannelValue = pixels[flipTables[flip][sector][px]][ch];
					sectorPixels[flip][sector][px][ch] = pixelChannelValue;
                    sectorPreWeightedPixels[flip][sector][px][ch] = preWeightedPixels[flipTables[flip][sector][px]][ch];
					sectorCumulative[flip][sector][ch] = sectorCumulative[flip][sector][ch] + pixelChannelValue;
				}
			}
		}
	}

	static const MSInt16 modifierTables[8][4] =
	{
		{ ParallelMath::MakeSInt16(-8), ParallelMath::MakeSInt16(-2), ParallelMath::MakeSInt16(2), ParallelMath::MakeSInt16(8) },
		{ ParallelMath::MakeSInt16(-17), ParallelMath::MakeSInt16(-5), ParallelMath::MakeSInt16(5), ParallelMath::MakeSInt16(17) },
		{ ParallelMath::MakeSInt16(-29), ParallelMath::MakeSInt16(-9), ParallelMath::MakeSInt16(9), ParallelMath::MakeSInt16(29) },
		{ ParallelMath::MakeSInt16(-42), ParallelMath::MakeSInt16(-13), ParallelMath::MakeSInt16(13), ParallelMath::MakeSInt16(42) },
		{ ParallelMath::MakeSInt16(-60), ParallelMath::MakeSInt16(-18), ParallelMath::MakeSInt16(18), ParallelMath::MakeSInt16(60) },
		{ ParallelMath::MakeSInt16(-80), ParallelMath::MakeSInt16(-24), ParallelMath::MakeSInt16(24), ParallelMath::MakeSInt16(80) },
		{ ParallelMath::MakeSInt16(-106), ParallelMath::MakeSInt16(-33), ParallelMath::MakeSInt16(33), ParallelMath::MakeSInt16(106) },
		{ ParallelMath::MakeSInt16(-183), ParallelMath::MakeSInt16(-47), ParallelMath::MakeSInt16(47), ParallelMath::MakeSInt16(183) },
	};

    bool isFakeBT709 = ((options.flags & cvtt::Flags::ETC_UseFakeBT709) != 0);

	for (int flip = 0; flip < 2; flip++)
	{
		drs.diffNumAttempts[0] = drs.diffNumAttempts[1] = zeroU15;

		MFloat bestIndError[2] = { ParallelMath::MakeFloat(FLT_MAX), ParallelMath::MakeFloat(FLT_MAX) };
		MUInt16 bestIndSelectors[2] = { ParallelMath::MakeUInt16(0), ParallelMath::MakeUInt16(0) };
		MUInt15 bestIndColors[2] = { zeroU15, zeroU15 };
		MUInt15 bestIndTable[2] = { zeroU15, zeroU15 };

		for (int d = 0; d < 2; d++)
		{
			for (int sector = 0; sector < 2; sector++)
			{
				const int16_t *potentialOffsets = cvtt::Tables::ETC1::g_potentialOffsets4;

				for (int table = 0; table < 8; table++)
				{
					int16_t numOffsets = *potentialOffsets++;

					MUInt15 possibleColors[cvtt::Tables::ETC1::g_maxPotentialOffsets];

                    MUInt15 quantized[3];
                    for (int oi = 0; oi < numOffsets; oi++)
                    {
                        if (!isFakeBT709)
                        {
						    for (int ch = 0; ch < 3; ch++)
						    {
                                // cu is in range 0..2040
                                MUInt15 cu15 = ParallelMath::Min(
                                    ParallelMath::MakeUInt15(2040),
                                    ParallelMath::ToUInt15(
                                        ParallelMath::Max(
                                            ParallelMath::MakeSInt16(0),
                                            ParallelMath::LosslessCast<MSInt16>::Cast(sectorCumulative[flip][sector][ch]) + ParallelMath::MakeSInt16(potentialOffsets[oi])
                                        )
                                    )
                                );

                                if (d == 1)
                                {
                                    //quantized[ch] = (cu * 31 + (cu >> 3) + 1024) >> 11;
                                    quantized[ch] = ParallelMath::ToUInt15(
                                        ParallelMath::RightShift(
                                            (ParallelMath::LosslessCast<MUInt16>::Cast(cu15) << 5) - ParallelMath::LosslessCast<MUInt16>::Cast(cu15) + ParallelMath::LosslessCast<MUInt16>::Cast(ParallelMath::RightShift(cu15, 3)) + ParallelMath::MakeUInt16(1024)
                                            , 11)
                                        );
                                }
                                else
                                {
                                    //quantized[ch] = (cu * 30 + (cu >> 3) + 2048) >> 12;
                                    quantized[ch] = ParallelMath::ToUInt15(
                                        ParallelMath::RightShift(
                                        (ParallelMath::LosslessCast<MUInt16>::Cast(cu15) << 5) - ParallelMath::LosslessCast<MUInt16>::Cast(cu15 << 1) + ParallelMath::LosslessCast<MUInt16>::Cast(ParallelMath::RightShift(cu15, 3)) + ParallelMath::MakeUInt16(2048)
                                            , 12)
                                    );
                                }
						    }
                        }
                        else
                        {
                            MUInt15 offsetCumulative[3];
						    for (int ch = 0; ch < 3; ch++)
						    {
                                // cu is in range 0..2040
                                MUInt15 cu15 = ParallelMath::Min(
                                    ParallelMath::MakeUInt15(2040),
                                    ParallelMath::ToUInt15(
                                        ParallelMath::Max(
                                            ParallelMath::MakeSInt16(0),
                                            ParallelMath::LosslessCast<MSInt16>::Cast(sectorCumulative[flip][sector][ch]) + ParallelMath::MakeSInt16(potentialOffsets[oi])
                                        )
                                    )
                                );

                                offsetCumulative[ch] = cu15;
						    }

                            if ((options.flags & cvtt::Flags::ETC_FakeBT709Accurate) != 0)
                                ResolveHalfBlockFakeBT709RoundingAccurate(quantized, offsetCumulative, d == 1);
                            else
                                ResolveHalfBlockFakeBT709RoundingFast(quantized, offsetCumulative, d == 1);
                        }

						possibleColors[oi] = quantized[0] | (quantized[1] << 5) | (quantized[2] << 10);
					}

					potentialOffsets += numOffsets;

                    ParallelMath::UInt15 numUniqueColors;
                    for (int block = 0; block < ParallelMath::ParallelSize; block++)
                    {
                        uint16_t blockNumUniqueColors = 1;
                        for (int i = 1; i < numOffsets; i++)
                        {
                            uint16_t color = ParallelMath::Extract(possibleColors[i], block);
                            if (color != ParallelMath::Extract(possibleColors[blockNumUniqueColors - 1], block))
                                ParallelMath::PutUInt15(possibleColors[blockNumUniqueColors++], block, color);
                        }

                        ParallelMath::PutUInt15(numUniqueColors, block, blockNumUniqueColors);
                    }

                    int maxUniqueColors = ParallelMath::Extract(numUniqueColors, 0);
                    for (int block = 1; block < ParallelMath::ParallelSize; block++)
                        maxUniqueColors = std::max<int>(maxUniqueColors, ParallelMath::Extract(numUniqueColors, block));

                    for (int block = 0; block < ParallelMath::ParallelSize; block++)
                    {
                        uint16_t fillColor = ParallelMath::Extract(possibleColors[0], block);
                        for (int i = ParallelMath::Extract(numUniqueColors, block); i < maxUniqueColors; i++)
                            ParallelMath::PutUInt15(possibleColors[i], block, fillColor);
                    }

					for (int i = 0; i < maxUniqueColors; i++)
					{
						MFloat error = ParallelMath::MakeFloatZero();
						MUInt16 selectors = ParallelMath::MakeUInt16(0);
                        MUInt15 quantized = possibleColors[i];
						TestHalfBlock(error, selectors, quantized, sectorPixels[flip][sector], sectorPreWeightedPixels[flip][sector], modifierTables[table], d == 1, options);

						if (d == 0)
						{
                            ParallelMath::Int16CompFlag errorBetter = ParallelMath::FloatFlagToInt16(ParallelMath::Less(error, bestIndError[sector]));
							if (ParallelMath::AnySet(errorBetter))
							{
								bestIndError[sector] = ParallelMath::Min(error, bestIndError[sector]);
								ParallelMath::ConditionalSet(bestIndSelectors[sector], errorBetter, selectors);
                                ParallelMath::ConditionalSet(bestIndColors[sector], errorBetter, quantized);
                                ParallelMath::ConditionalSet(bestIndTable[sector], errorBetter, ParallelMath::MakeUInt15(table));
							}
						}
						else
						{
                            ParallelMath::Int16CompFlag isInBounds = ParallelMath::Less(ParallelMath::MakeUInt15(i), numUniqueColors);

							MUInt15 storageIndexes = drs.diffNumAttempts[sector];
                            drs.diffNumAttempts[sector] = drs.diffNumAttempts[sector] + ParallelMath::SelectOrZero(isInBounds, ParallelMath::MakeUInt15(1));

                            for (int block = 0; block < ParallelMath::ParallelSize; block++)
                            {
                                int storageIndex = ParallelMath::Extract(storageIndexes, block);

                                ParallelMath::PutFloat(drs.diffErrors[sector][storageIndex], block, ParallelMath::Extract(error, block));
                                ParallelMath::PutUInt16(drs.diffSelectors[sector][storageIndex], block, ParallelMath::Extract(selectors, block));
                                ParallelMath::PutUInt15(drs.diffColors[sector][storageIndex], block, ParallelMath::Extract(quantized, block));
                                ParallelMath::PutUInt15(drs.diffTables[sector][storageIndex], block, table);
                            }
						}
					}
				}
			}

			if (d == 0)
			{
				MFloat bestIndErrorTotal = bestIndError[0] + bestIndError[1];
                ParallelMath::Int16CompFlag errorBetter = ParallelMath::FloatFlagToInt16(ParallelMath::Less(bestIndErrorTotal, bestTotalError));
				if (ParallelMath::AnySet(errorBetter))
				{
                    bestIsThisMode = bestIsThisMode | errorBetter;

					bestTotalError = ParallelMath::Min(bestTotalError, bestIndErrorTotal);
					ParallelMath::ConditionalSet(bestFlip, errorBetter, ParallelMath::MakeUInt15(flip));
                    ParallelMath::ConditionalSet(bestD, errorBetter, ParallelMath::MakeUInt15(d));
					for (int sector = 0; sector < 2; sector++)
					{
                        ParallelMath::ConditionalSet(bestColors[sector], errorBetter, bestIndColors[sector]);
                        ParallelMath::ConditionalSet(bestSelectors[sector], errorBetter, bestIndSelectors[sector]);
                        ParallelMath::ConditionalSet(bestTables[sector], errorBetter, bestIndTable[sector]);
					}
				}
			}
			else
			{
                // Differential
                // We do this part scalar because most of the cost benefit of parallelization is in error evaluation,
                // and this code has a LOT of early-outs and disjointed index lookups that vary heavily between blocks
                // and save a lot of time.
                for (int block = 0; block < ParallelMath::ParallelSize; block++)
                {
                    float blockBestTotalError = ParallelMath::Extract(bestTotalError, block);
                    float bestDiffErrors[2] = { FLT_MAX, FLT_MAX };
                    uint16_t bestDiffSelectors[2] = { 0, 0 };
                    uint16_t bestDiffColors[2] = { 0, 0 };
                    uint16_t bestDiffTables[2] = { 0, 0 };
                    for (int sector = 0; sector < 2; sector++)
                    {
                        unsigned int sectorNumAttempts = ParallelMath::Extract(drs.diffNumAttempts[sector], block);
                        for (unsigned int i = 0; i < sectorNumAttempts; i++)
                        {
                            float error = ParallelMath::Extract(drs.diffErrors[sector][i], block);
                            if (error < bestDiffErrors[sector])
                            {
                                bestDiffErrors[sector] = error;
                                bestDiffSelectors[sector] = ParallelMath::Extract(drs.diffSelectors[sector][i], block);
                                bestDiffColors[sector] = ParallelMath::Extract(drs.diffColors[sector][i], block);
                                bestDiffTables[sector] = ParallelMath::Extract(drs.diffTables[sector][i], block);
                            }
                        }
                    }

                    // The best differential possibilities must be better than the best total error
                    if (bestDiffErrors[0] + bestDiffErrors[1] < blockBestTotalError)
                    {
                        // Fast path if the best possible case is legal
                        if (ETCDifferentialIsLegalScalar(bestDiffColors[0], bestDiffColors[1]))
                        {
                            ParallelMath::PutBoolInt16(bestIsThisMode, block, true);
                            ParallelMath::PutFloat(bestTotalError, block, bestDiffErrors[0] + bestDiffErrors[1]);
                            ParallelMath::PutUInt15(bestFlip, block, flip);
                            ParallelMath::PutUInt15(bestD, block, d);
                            for (int sector = 0; sector < 2; sector++)
                            {
                                ParallelMath::PutUInt15(bestColors[sector], block, bestDiffColors[sector]);
                                ParallelMath::PutUInt16(bestSelectors[sector], block, bestDiffSelectors[sector]);
                                ParallelMath::PutUInt15(bestTables[sector], block, bestDiffTables[sector]);
                            }
                        }
                        else
                        {
                            // Slow path: Sort the possible cases by quality, and search valid combinations
                            // TODO: Pre-flatten the error lists so this is nicer to cache
                            unsigned int numSortIndexes[2] = { 0, 0 };
                            for (int sector = 0; sector < 2; sector++)
                            {
                                unsigned int sectorNumAttempts = ParallelMath::Extract(drs.diffNumAttempts[sector], block);

                                for (unsigned int i = 0; i < sectorNumAttempts; i++)
                                {
                                    if (ParallelMath::Extract(drs.diffErrors[sector][i], block) < blockBestTotalError)
                                        drs.attemptSortIndexes[sector][numSortIndexes[sector]++] = i;
                                }

                                struct SortPredicate
                                {
                                    const MFloat *diffErrors;
                                    int block;

                                    bool operator()(uint16_t a, uint16_t b) const
                                    {
                                        float errorA = ParallelMath::Extract(diffErrors[a], block);
                                        float errorB = ParallelMath::Extract(diffErrors[b], block);

                                        if (errorA < errorB)
                                            return true;
                                        if (errorA > errorB)
                                            return false;

                                        return a < b;
                                    }
                                };

                                SortPredicate sp;
                                sp.diffErrors = drs.diffErrors[sector];
                                sp.block = block;

                                std::sort<uint16_t*, const SortPredicate&>(drs.attemptSortIndexes[sector], drs.attemptSortIndexes[sector] + numSortIndexes[sector], sp);
                            }

                            int scannedElements = 0;
                            for (unsigned int i = 0; i < numSortIndexes[0]; i++)
                            {
                                unsigned int attemptIndex0 = drs.attemptSortIndexes[0][i];
                                float error0 = ParallelMath::Extract(drs.diffErrors[0][attemptIndex0], block);

                                scannedElements++;

                                if (error0 >= blockBestTotalError)
                                    break;

                                float maxError1 = ParallelMath::Extract(bestTotalError, block) - error0;
                                uint16_t diffColor0 = ParallelMath::Extract(drs.diffColors[0][attemptIndex0], block);

                                if (maxError1 < bestDiffErrors[1])
                                    break;

                                for (unsigned int j = 0; j < numSortIndexes[1]; j++)
                                {
                                    unsigned int attemptIndex1 = drs.attemptSortIndexes[1][j];
                                    float error1 = ParallelMath::Extract(drs.diffErrors[1][attemptIndex1], block);

                                    scannedElements++;

                                    if (error1 >= maxError1)
                                        break;

                                    uint16_t diffColor1 = ParallelMath::Extract(drs.diffColors[1][attemptIndex1], block);

                                    if (ETCDifferentialIsLegalScalar(diffColor0, diffColor1))
                                    {
                                        blockBestTotalError = error0 + error1;

                                        ParallelMath::PutBoolInt16(bestIsThisMode, block, true);
                                        ParallelMath::PutFloat(bestTotalError, block, blockBestTotalError);
                                        ParallelMath::PutUInt15(bestFlip, block, flip);
                                        ParallelMath::PutUInt15(bestD, block, d);
                                        ParallelMath::PutUInt15(bestColors[0], block, diffColor0);
                                        ParallelMath::PutUInt15(bestColors[1], block, diffColor1);
                                        ParallelMath::PutUInt16(bestSelectors[0], block, ParallelMath::Extract(drs.diffSelectors[0][attemptIndex0], block));
                                        ParallelMath::PutUInt16(bestSelectors[1], block, ParallelMath::Extract(drs.diffSelectors[1][attemptIndex1], block));
                                        ParallelMath::PutUInt15(bestTables[0], block, ParallelMath::Extract(drs.diffTables[0][attemptIndex0], block));
                                        ParallelMath::PutUInt15(bestTables[1], block, ParallelMath::Extract(drs.diffTables[1][attemptIndex1], block));
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
			}
		}
	}

    for (int block = 0; block < ParallelMath::ParallelSize; block++)
    {
        if (!ParallelMath::Extract(bestIsThisMode, block))
            continue;

        uint32_t highBits = 0;
        uint32_t lowBits = 0;

        int blockBestFlip = ParallelMath::Extract(bestFlip, block);
        int blockBestD = ParallelMath::Extract(bestD, block);

        int colors[2][3];
        for (int sector = 0; sector < 2; sector++)
        {
            int sectorColor = ParallelMath::Extract(bestColors[sector], block);
            for (int ch = 0; ch < 3; ch++)
                colors[sector][ch] = (sectorColor >> (ch * 5)) & 31;
        }
        if (blockBestD == 0)
        {
            highBits |= colors[0][0] << 28;
            highBits |= colors[1][0] << 24;
            highBits |= colors[0][1] << 20;
            highBits |= colors[1][1] << 16;
            highBits |= colors[0][2] << 12;
            highBits |= colors[1][2] << 8;
        }
        else
        {
            highBits |= colors[0][0] << 27;
            highBits |= ((colors[1][0] - colors[0][0]) & 7) << 24;
            highBits |= colors[0][1] << 19;
            highBits |= ((colors[1][1] - colors[0][1]) & 7) << 16;
            highBits |= colors[0][2] << 11;
            highBits |= ((colors[1][2] - colors[0][2]) & 7) << 8;
        }
        highBits |= (ParallelMath::Extract(bestTables[0], block) << 5);
        highBits |= (ParallelMath::Extract(bestTables[1], block) << 2);
        highBits |= (blockBestD << 1);
        highBits |= blockBestFlip;

        const uint8_t modifierCodes[4] = { 3, 2, 0, 1 };

        uint8_t unpackedSelectors[16];
        uint8_t unpackedSelectorCodes[16];
        for (int sector = 0; sector < 2; sector++)
        {
            int blockSectorBestSelectors = ParallelMath::Extract(bestSelectors[sector], block);

            for (int px = 0; px < 8; px++)
            {
                int selector = (blockSectorBestSelectors >> (2 * px)) & 3;
                unpackedSelectorCodes[flipTables[blockBestFlip][sector][px]] = modifierCodes[selector];
                unpackedSelectors[flipTables[blockBestFlip][sector][px]] = selector;
            }
        }

        const int pixelSelectorOrder[16] = { 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15 };

        int lowBitOffset = 0;
        for (int sb = 0; sb < 2; sb++)
            for (int px = 0; px < 16; px++)
                lowBits |= ((unpackedSelectorCodes[pixelSelectorOrder[px]] >> sb) & 1) << (px + sb * 16);

        for (int i = 0; i < 4; i++)
            outputBuffer[block * 8 + i] = (highBits >> (24 - i * 8)) & 0xff;
        for (int i = 0; i < 4; i++)
            outputBuffer[block * 8 + i + 4] = (lowBits >> (24 - i * 8)) & 0xff;
    }
}

cvtt::ETC1CompressionData *cvtt::Internal::ETCComputer::AllocETC1Data(cvtt::Kernels::allocFunc_t allocFunc, void *context)
{
    void *buffer = allocFunc(context, sizeof(cvtt::Internal::ETCComputer::ETC1CompressionDataInternal));
    if (!buffer)
        return NULL;
    new (buffer) cvtt::Internal::ETCComputer::ETC1CompressionDataInternal(context);
    return static_cast<ETC1CompressionData*>(buffer);
}

void cvtt::Internal::ETCComputer::ReleaseETC1Data(ETC1CompressionData *compressionData, cvtt::Kernels::freeFunc_t freeFunc)
{
    cvtt::Internal::ETCComputer::ETC1CompressionDataInternal* internalData = static_cast<cvtt::Internal::ETCComputer::ETC1CompressionDataInternal*>(compressionData);
    void *context = internalData->m_context;
    internalData->~ETC1CompressionDataInternal();
    freeFunc(context, compressionData, sizeof(cvtt::Internal::ETCComputer::ETC1CompressionDataInternal));
}

cvtt::ETC2CompressionData *cvtt::Internal::ETCComputer::AllocETC2Data(cvtt::Kernels::allocFunc_t allocFunc, void *context)
{
    void *buffer = allocFunc(context, sizeof(cvtt::Internal::ETCComputer::ETC2CompressionDataInternal));
    if (!buffer)
        return NULL;
    new (buffer) cvtt::Internal::ETCComputer::ETC2CompressionDataInternal(context);
    return static_cast<ETC2CompressionData*>(buffer);
}

void cvtt::Internal::ETCComputer::ReleaseETC2Data(ETC2CompressionData *compressionData, cvtt::Kernels::freeFunc_t freeFunc)
{
    cvtt::Internal::ETCComputer::ETC2CompressionDataInternal* internalData = static_cast<cvtt::Internal::ETCComputer::ETC2CompressionDataInternal*>(compressionData);
    void *context = internalData->m_context;
    internalData->~ETC2CompressionDataInternal();
    freeFunc(context, compressionData, sizeof(cvtt::Internal::ETCComputer::ETC2CompressionDataInternal));
}

#endif
