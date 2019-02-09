#pragma once
#ifndef __CVTT_CONVECTIONKERNELS_ETC_H__
#define __CVTT_CONVECTIONKERNELS_ETC_H__

#include "ConvectionKernels.h"
#include "ConvectionKernels_ParallelMath.h"

namespace cvtt
{
    namespace Internal
    {
        class ETCComputer
        {
        public:
            static void CompressETC1Block(uint8_t *outputBuffer, const PixelBlockU8 *inputBlocks, ETC1CompressionData *compressionData);
            static void CompressETC2Block(uint8_t *outputBuffer, const PixelBlockU8 *pixelBlocks, ETC2CompressionData *compressionData);

            static ETC2CompressionData *AllocETC2Data(cvtt::Kernels::allocFunc_t allocFunc, void *context);
            static void ReleaseETC2Data(ETC2CompressionData *compressionData, cvtt::Kernels::freeFunc_t freeFunc);

            static ETC1CompressionData *AllocETC1Data(cvtt::Kernels::allocFunc_t allocFunc, void *context);
            static void ReleaseETC1Data(ETC1CompressionData *compressionData, cvtt::Kernels::freeFunc_t freeFunc);

        private:
            typedef ParallelMath::Float MFloat;
            typedef ParallelMath::SInt16 MSInt16;
            typedef ParallelMath::UInt15 MUInt15;
            typedef ParallelMath::UInt16 MUInt16;
            typedef ParallelMath::SInt32 MSInt32;

            struct DifferentialResolveStorage
            {
                static const unsigned int MaxAttemptsPerSector = 57 + 81 + 81 + 81 + 81 + 81 + 81 + 81;

                MUInt15 diffNumAttempts[2];
                MFloat diffErrors[2][MaxAttemptsPerSector];
                MUInt16 diffSelectors[2][MaxAttemptsPerSector];
                MUInt15 diffColors[2][MaxAttemptsPerSector];
                MUInt15 diffTables[2][MaxAttemptsPerSector];

                uint16_t attemptSortIndexes[2][MaxAttemptsPerSector];
            };

            struct HModeEval
            {
                MFloat errors[62][16];
                MUInt16 signBits[62];
                MUInt15 uniqueQuantizedColors[62];
                MUInt15 numUniqueColors[2];
            };

            struct ETC1CompressionDataInternal : public cvtt::ETC1CompressionData
            {
                explicit ETC1CompressionDataInternal(void *context)
                    : m_context(context)
                {
                }

                DifferentialResolveStorage m_drs;
                void *m_context;
            };

            struct ETC2CompressionDataInternal : public cvtt::ETC2CompressionData
            {
                explicit ETC2CompressionDataInternal(void *context)
                    : m_context(context)
                {
                }

                HModeEval m_h;
                DifferentialResolveStorage m_drs;

                void *m_context;
            };

            static MFloat ComputeError(const MUInt15 pixelA[3], const MUInt15 pixelB[3]);
            static void TestHalfBlock(MFloat &outError, MUInt16 &outSelectors, MUInt15 quantizedPackedColor, const MUInt15 pixels[8][3], const MSInt16 modifiers[4], bool isDifferential);

            static ParallelMath::Int16CompFlag ETCDifferentialIsLegalForChannel(const MUInt15 &a, const MUInt15 &b);
            static ParallelMath::Int16CompFlag ETCDifferentialIsLegal(const MUInt15 &a, const MUInt15 &b);
            static bool ETCDifferentialIsLegalForChannelScalar(const uint16_t &a, const uint16_t &b);
            static bool ETCDifferentialIsLegalScalar(const uint16_t &a, const uint16_t &b);
            static void EncodeTMode(uint8_t *outputBuffer, MFloat &bestError, const ParallelMath::Int16CompFlag isIsolated[16], const MUInt15 pixels[16][3]);
            static void EncodeHMode(uint8_t *outputBuffer, MFloat &bestError, const ParallelMath::Int16CompFlag groupings[16], const MUInt15 pixels[16][3], HModeEval &he);
            static MUInt15 DecodePlanarCoeff(const MUInt15 &coeff, int ch);
            static void EncodePlanar(uint8_t *outputBuffer, MFloat &bestError, const MUInt15 pixels[16][3]);

            static void CompressETC1BlockInternal(MFloat &bestTotalError, uint8_t *outputBuffer, const PixelBlockU8 *inputBlocks, DifferentialResolveStorage& compressionData);
        };
    }
}

#endif
