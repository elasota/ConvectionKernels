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

#include "ConvectionKernels_BC67.h"

#include "ConvectionKernels_AggregatedError.h"
#include "ConvectionKernels_BCCommon.h"
#include "ConvectionKernels_BC7_Prio.h"
#include "ConvectionKernels_BC7_SingleColor.h"
#include "ConvectionKernels_BC6H_IO.h"
#include "ConvectionKernels_EndpointRefiner.h"
#include "ConvectionKernels_EndpointSelector.h"
#include "ConvectionKernels_IndexSelectorHDR.h"
#include "ConvectionKernels_ParallelMath.h"
#include "ConvectionKernels_UnfinishedEndpoints.h"

namespace cvtt
{
    namespace Internal
    {
        namespace BC67
        {
            typedef ParallelMath::Float MFloat;
            typedef ParallelMath::UInt15 MUInt15;

            struct WorkInfo
            {
                MUInt15 m_mode;
                MFloat m_error;
                MUInt15 m_ep[3][2][4];
                MUInt15 m_indexes[16];
                MUInt15 m_indexes2[16];

                union
                {
                    MUInt15 m_partition;
                    struct IndexSelectorAndRotation
                    {
                        MUInt15 m_indexSelector;
                        MUInt15 m_rotation;
                    } m_isr;
                } m_u;
            };
        }

        namespace BC7Data
        {
            enum AlphaMode
            {
                AlphaMode_Combined,
                AlphaMode_Separate,
                AlphaMode_None,
            };

            enum PBitMode
            {
                PBitMode_PerEndpoint,
                PBitMode_PerSubset,
                PBitMode_None
            };

            struct BC7ModeInfo
            {
                PBitMode m_pBitMode;
                AlphaMode m_alphaMode;
                int m_rgbBits;
                int m_alphaBits;
                int m_partitionBits;
                int m_numSubsets;
                int m_indexBits;
                int m_alphaIndexBits;
                bool m_hasIndexSelector;
            };

            BC7ModeInfo g_modes[] =
            {
                { PBitMode_PerEndpoint, AlphaMode_None, 4, 0, 4, 3, 3, 0, false },     // 0
                { PBitMode_PerSubset, AlphaMode_None, 6, 0, 6, 2, 3, 0, false },       // 1
                { PBitMode_None, AlphaMode_None, 5, 0, 6, 3, 2, 0, false },            // 2
                { PBitMode_PerEndpoint, AlphaMode_None, 7, 0, 6, 2, 2, 0, false },     // 3 (Mode reference has an error, P-bit is really per-endpoint)

                { PBitMode_None, AlphaMode_Separate, 5, 6, 0, 1, 2, 3, true },         // 4
                { PBitMode_None, AlphaMode_Separate, 7, 8, 0, 1, 2, 2, false },        // 5
                { PBitMode_PerEndpoint, AlphaMode_Combined, 7, 7, 0, 1, 4, 0, false }, // 6
                { PBitMode_PerEndpoint, AlphaMode_Combined, 5, 5, 6, 2, 2, 0, false }  // 7
            };

            const int g_weight2[] = { 0, 21, 43, 64 };
            const int g_weight3[] = { 0, 9, 18, 27, 37, 46, 55, 64 };
            const int g_weight4[] = { 0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64 };

            const int *g_weightTables[] =
            {
                NULL,
                NULL,
                g_weight2,
                g_weight3,
                g_weight4
            };

            struct BC6HModeInfo
            {
                uint16_t m_modeID;
                bool m_partitioned;
                bool m_transformed;
                int m_aPrec;
                int m_bPrec[3];
            };

            // [partitioned][precision]
            bool g_hdrModesExistForPrecision[2][17] =
            {
                //0      1      2      3      4      5      6      7      8      9      10     11     12     13     14     15     16
                { false, false, false, false, false, false, false, false, false, false, true,  true,  true,  false, false, false, true },
                { false, false, false, false, false, false, true,  true,  true,  true,  true,  true,  false, false, false, false, false },
            };

            BC6HModeInfo g_hdrModes[] =
            {
                { 0x00, true,  true,  10,{ 5, 5, 5 } },
                { 0x01, true,  true,  7,{ 6, 6, 6 } },
                { 0x02, true,  true,  11,{ 5, 4, 4 } },
                { 0x06, true,  true,  11,{ 4, 5, 4 } },
                { 0x0a, true,  true,  11,{ 4, 4, 5 } },
                { 0x0e, true,  true,  9,{ 5, 5, 5 } },
                { 0x12, true,  true,  8,{ 6, 5, 5 } },
                { 0x16, true,  true,  8,{ 5, 6, 5 } },
                { 0x1a, true,  true,  8,{ 5, 5, 6 } },
                { 0x1e, true,  false, 6,{ 6, 6, 6 } },
                { 0x03, false, false, 10,{ 10, 10, 10 } },
                { 0x07, false, true,  11,{ 9, 9, 9 } },
                { 0x0b, false, true,  12,{ 8, 8, 8 } },
                { 0x0f, false, true,  16,{ 4, 4, 4 } },
            };

            const int g_maxHDRPrecision = 16;

            static const size_t g_numHDRModes = sizeof(g_hdrModes) / sizeof(g_hdrModes[0]);

            static uint16_t g_partitionMap[64] =
            {
                0xCCCC, 0x8888, 0xEEEE, 0xECC8,
                0xC880, 0xFEEC, 0xFEC8, 0xEC80,
                0xC800, 0xFFEC, 0xFE80, 0xE800,
                0xFFE8, 0xFF00, 0xFFF0, 0xF000,
                0xF710, 0x008E, 0x7100, 0x08CE,
                0x008C, 0x7310, 0x3100, 0x8CCE,
                0x088C, 0x3110, 0x6666, 0x366C,
                0x17E8, 0x0FF0, 0x718E, 0x399C,
                0xaaaa, 0xf0f0, 0x5a5a, 0x33cc,
                0x3c3c, 0x55aa, 0x9696, 0xa55a,
                0x73ce, 0x13c8, 0x324c, 0x3bdc,
                0x6996, 0xc33c, 0x9966, 0x660,
                0x272, 0x4e4, 0x4e40, 0x2720,
                0xc936, 0x936c, 0x39c6, 0x639c,
                0x9336, 0x9cc6, 0x817e, 0xe718,
                0xccf0, 0xfcc, 0x7744, 0xee22,
            };

            static uint32_t g_partitionMap2[64] =
            {
                0xaa685050, 0x6a5a5040, 0x5a5a4200, 0x5450a0a8,
                0xa5a50000, 0xa0a05050, 0x5555a0a0, 0x5a5a5050,
                0xaa550000, 0xaa555500, 0xaaaa5500, 0x90909090,
                0x94949494, 0xa4a4a4a4, 0xa9a59450, 0x2a0a4250,
                0xa5945040, 0x0a425054, 0xa5a5a500, 0x55a0a0a0,
                0xa8a85454, 0x6a6a4040, 0xa4a45000, 0x1a1a0500,
                0x0050a4a4, 0xaaa59090, 0x14696914, 0x69691400,
                0xa08585a0, 0xaa821414, 0x50a4a450, 0x6a5a0200,
                0xa9a58000, 0x5090a0a8, 0xa8a09050, 0x24242424,
                0x00aa5500, 0x24924924, 0x24499224, 0x50a50a50,
                0x500aa550, 0xaaaa4444, 0x66660000, 0xa5a0a5a0,
                0x50a050a0, 0x69286928, 0x44aaaa44, 0x66666600,
                0xaa444444, 0x54a854a8, 0x95809580, 0x96969600,
                0xa85454a8, 0x80959580, 0xaa141414, 0x96960000,
                0xaaaa1414, 0xa05050a0, 0xa0a5a5a0, 0x96000000,
                0x40804080, 0xa9a8a9a8, 0xaaaaaa44, 0x2a4a5254,
            };

            static int g_fixupIndexes2[64] =
            {
                15,15,15,15,
                15,15,15,15,
                15,15,15,15,
                15,15,15,15,
                15, 2, 8, 2,
                2, 8, 8,15,
                2, 8, 2, 2,
                8, 8, 2, 2,

                15,15, 6, 8,
                2, 8,15,15,
                2, 8, 2, 2,
                2,15,15, 6,
                6, 2, 6, 8,
                15,15, 2, 2,
                15,15,15,15,
                15, 2, 2,15,
            };

            static int g_fixupIndexes3[64][2] =
            {
                { 3,15 },{ 3, 8 },{ 15, 8 },{ 15, 3 },
                { 8,15 },{ 3,15 },{ 15, 3 },{ 15, 8 },
                { 8,15 },{ 8,15 },{ 6,15 },{ 6,15 },
                { 6,15 },{ 5,15 },{ 3,15 },{ 3, 8 },
                { 3,15 },{ 3, 8 },{ 8,15 },{ 15, 3 },
                { 3,15 },{ 3, 8 },{ 6,15 },{ 10, 8 },
                { 5, 3 },{ 8,15 },{ 8, 6 },{ 6,10 },
                { 8,15 },{ 5,15 },{ 15,10 },{ 15, 8 },

                { 8,15 },{ 15, 3 },{ 3,15 },{ 5,10 },
                { 6,10 },{ 10, 8 },{ 8, 9 },{ 15,10 },
                { 15, 6 },{ 3,15 },{ 15, 8 },{ 5,15 },
                { 15, 3 },{ 15, 6 },{ 15, 6 },{ 15, 8 },
                { 3,15 },{ 15, 3 },{ 5,15 },{ 5,15 },
                { 5,15 },{ 8,15 },{ 5,15 },{ 10,15 },
                { 5,15 },{ 10,15 },{ 8,15 },{ 13,15 },
                { 15, 3 },{ 12,15 },{ 3,15 },{ 3, 8 },
            };

            static const unsigned char g_fragments[] =
            {
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,  // 0, 16
                0, 1, 2, 3,  // 16, 4
                0, 1, 4,  // 20, 3
                0, 1, 2, 4,  // 23, 4
                2, 3, 7,  // 27, 3
                1, 2, 3, 7,  // 30, 4
                0, 1, 2, 3, 4, 5, 6, 7,  // 34, 8
                0, 1, 4, 8,  // 42, 4
                0, 1, 2, 4, 5, 8,  // 46, 6
                0, 1, 2, 3, 4, 5, 6, 8,  // 52, 8
                1, 4, 5, 6, 9,  // 60, 5
                2, 5, 6, 7, 10,  // 65, 5
                5, 6, 9, 10,  // 70, 4
                2, 3, 7, 11,  // 74, 4
                1, 2, 3, 6, 7, 11,  // 78, 6
                0, 1, 2, 3, 5, 6, 7, 11,  // 84, 8
                0, 1, 2, 3, 8, 9, 10, 11,  // 92, 8
                2, 3, 6, 7, 8, 9, 10, 11,  // 100, 8
                4, 5, 6, 7, 8, 9, 10, 11,  // 108, 8
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,  // 116, 12
                0, 4, 8, 12,  // 128, 4
                0, 2, 3, 4, 6, 7, 8, 12,  // 132, 8
                0, 1, 2, 4, 5, 8, 9, 12,  // 140, 8
                0, 1, 2, 3, 4, 5, 6, 8, 9, 12,  // 148, 10
                3, 6, 7, 8, 9, 12,  // 158, 6
                3, 5, 6, 7, 8, 9, 10, 12,  // 164, 8
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12,  // 172, 12
                0, 1, 2, 5, 6, 7, 11, 12,  // 184, 8
                5, 8, 9, 10, 13,  // 192, 5
                8, 12, 13,  // 197, 3
                4, 8, 12, 13,  // 200, 4
                2, 3, 6, 9, 12, 13,  // 204, 6
                0, 1, 2, 3, 8, 9, 12, 13,  // 210, 8
                0, 1, 4, 5, 8, 9, 12, 13,  // 218, 8
                2, 3, 6, 7, 8, 9, 12, 13,  // 226, 8
                2, 3, 5, 6, 9, 10, 12, 13,  // 234, 8
                0, 3, 6, 7, 9, 10, 12, 13,  // 242, 8
                0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13,  // 250, 12
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13,  // 262, 13
                2, 3, 4, 7, 8, 11, 12, 13,  // 275, 8
                1, 2, 6, 7, 8, 11, 12, 13,  // 283, 8
                2, 3, 4, 6, 7, 8, 9, 11, 12, 13,  // 291, 10
                2, 3, 4, 5, 10, 11, 12, 13,  // 301, 8
                0, 1, 6, 7, 10, 11, 12, 13,  // 309, 8
                6, 9, 10, 11, 14,  // 317, 5
                0, 2, 4, 6, 8, 10, 12, 14,  // 322, 8
                1, 3, 5, 7, 8, 10, 12, 14,  // 330, 8
                1, 3, 4, 6, 9, 11, 12, 14,  // 338, 8
                0, 2, 5, 7, 9, 11, 12, 14,  // 346, 8
                0, 3, 4, 5, 8, 9, 13, 14,  // 354, 8
                2, 3, 4, 7, 8, 9, 13, 14,  // 362, 8
                1, 2, 5, 6, 9, 10, 13, 14,  // 370, 8
                0, 3, 4, 7, 9, 10, 13, 14,  // 378, 8
                0, 3, 5, 6, 8, 11, 13, 14,  // 386, 8
                1, 2, 4, 7, 8, 11, 13, 14,  // 394, 8
                0, 1, 4, 7, 10, 11, 13, 14,  // 402, 8
                0, 3, 6, 7, 10, 11, 13, 14,  // 410, 8
                8, 12, 13, 14,  // 418, 4
                1, 2, 3, 7, 8, 12, 13, 14,  // 422, 8
                4, 8, 9, 12, 13, 14,  // 430, 6
                0, 4, 5, 8, 9, 12, 13, 14,  // 436, 8
                1, 2, 3, 6, 7, 8, 9, 12, 13, 14,  // 444, 10
                2, 6, 8, 9, 10, 12, 13, 14,  // 454, 8
                0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14,  // 462, 12
                0, 7, 9, 10, 11, 12, 13, 14,  // 474, 8
                1, 2, 3, 4, 5, 6, 8, 15,  // 482, 8
                3, 7, 11, 15,  // 490, 4
                0, 1, 3, 4, 5, 7, 11, 15,  // 494, 8
                0, 4, 5, 10, 11, 15,  // 502, 6
                1, 2, 3, 6, 7, 10, 11, 15,  // 508, 8
                0, 1, 2, 3, 5, 6, 7, 10, 11, 15,  // 516, 10
                0, 4, 5, 6, 9, 10, 11, 15,  // 526, 8
                0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 15,  // 534, 12
                1, 2, 4, 5, 8, 9, 12, 15,  // 546, 8
                2, 3, 5, 6, 8, 9, 12, 15,  // 554, 8
                0, 3, 5, 6, 9, 10, 12, 15,  // 562, 8
                1, 2, 4, 7, 9, 10, 12, 15,  // 570, 8
                1, 2, 5, 6, 8, 11, 12, 15,  // 578, 8
                0, 3, 4, 7, 8, 11, 12, 15,  // 586, 8
                0, 1, 5, 6, 10, 11, 12, 15,  // 594, 8
                1, 2, 6, 7, 10, 11, 12, 15,  // 602, 8
                1, 3, 4, 6, 8, 10, 13, 15,  // 610, 8
                0, 2, 5, 7, 8, 10, 13, 15,  // 618, 8
                0, 2, 4, 6, 9, 11, 13, 15,  // 626, 8
                1, 3, 5, 7, 9, 11, 13, 15,  // 634, 8
                0, 1, 2, 3, 4, 5, 7, 8, 12, 13, 15,  // 642, 11
                2, 3, 4, 5, 8, 9, 14, 15,  // 653, 8
                0, 1, 6, 7, 8, 9, 14, 15,  // 661, 8
                0, 1, 5, 10, 14, 15,  // 669, 6
                0, 3, 4, 5, 9, 10, 14, 15,  // 675, 8
                0, 1, 5, 6, 9, 10, 14, 15,  // 683, 8
                11, 14, 15,  // 691, 3
                7, 11, 14, 15,  // 694, 4
                1, 2, 4, 5, 8, 11, 14, 15,  // 698, 8
                0, 1, 4, 7, 8, 11, 14, 15,  // 706, 8
                0, 1, 4, 5, 10, 11, 14, 15,  // 714, 8
                2, 3, 6, 7, 10, 11, 14, 15,  // 722, 8
                4, 5, 6, 7, 10, 11, 14, 15,  // 730, 8
                0, 1, 4, 5, 7, 8, 10, 11, 14, 15,  // 738, 10
                0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 14, 15,  // 748, 12
                0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 14, 15,  // 760, 13
                0, 1, 2, 3, 4, 6, 7, 11, 12, 14, 15,  // 773, 11
                3, 4, 8, 9, 10, 13, 14, 15,  // 784, 8
                11, 13, 14, 15,  // 792, 4
                0, 1, 2, 4, 11, 13, 14, 15,  // 796, 8
                0, 1, 2, 4, 5, 10, 11, 13, 14, 15,  // 804, 10
                7, 10, 11, 13, 14, 15,  // 814, 6
                3, 6, 7, 10, 11, 13, 14, 15,  // 820, 8
                1, 5, 9, 10, 11, 13, 14, 15,  // 828, 8
                1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15,  // 836, 12
                12, 13, 14, 15,  // 848, 4
                0, 1, 2, 3, 12, 13, 14, 15,  // 852, 8
                0, 1, 4, 5, 12, 13, 14, 15,  // 860, 8
                4, 5, 6, 7, 12, 13, 14, 15,  // 868, 8
                4, 8, 9, 10, 12, 13, 14, 15,  // 876, 8
                0, 4, 5, 8, 9, 10, 12, 13, 14, 15,  // 884, 10
                0, 1, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15,  // 894, 12
                0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15,  // 906, 12
                0, 1, 3, 4, 8, 9, 11, 12, 13, 14, 15,  // 918, 11
                0, 2, 3, 7, 8, 10, 11, 12, 13, 14, 15,  // 929, 11
                7, 9, 10, 11, 12, 13, 14, 15,  // 940, 8
                3, 6, 7, 9, 10, 11, 12, 13, 14, 15,  // 948, 10
                2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15,  // 958, 12
                8, 9, 10, 11, 12, 13, 14, 15,  // 970, 8
                0, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15,  // 978, 12
                0, 1, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15,  // 990, 13
                3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,  // 1003, 12
                2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,  // 1015, 13
                4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,  // 1028, 12
                0, 2,  // 1040, 2
                1, 3,  // 1042, 2
                0, 1, 4, 5,  // 1044, 4
                0, 1, 2, 4, 5,  // 1048, 5
                2, 3, 6,  // 1053, 3
                0, 2, 4, 6,  // 1056, 4
                1, 2, 5, 6,  // 1060, 4
                0, 1, 2, 3, 5, 6,  // 1064, 6
                0, 1, 2, 4, 5, 6,  // 1070, 6
                0, 1, 2, 3, 4, 5, 6,  // 1076, 7
                0, 3, 4, 7,  // 1083, 4
                0, 1, 2, 3, 4, 7,  // 1087, 6
                1, 3, 5, 7,  // 1093, 4
                2, 3, 6, 7,  // 1097, 4
                1, 2, 3, 6, 7,  // 1101, 5
                1, 2, 3, 5, 6, 7,  // 1106, 6
                0, 1, 2, 3, 5, 6, 7,  // 1112, 7
                4, 5, 6, 7,  // 1119, 4
                0, 8,  // 1123, 2
                0, 1, 4, 5, 8,  // 1125, 5
                0, 1, 8, 9,  // 1130, 4
                4, 5, 8, 9,  // 1134, 4
                0, 1, 4, 5, 8, 9,  // 1138, 6
                2, 6, 8, 9,  // 1144, 4
                6, 7, 8, 9,  // 1148, 4
                0, 2, 4, 6, 8, 10,  // 1152, 6
                1, 2, 5, 6, 9, 10,  // 1158, 6
                0, 3, 4, 7, 9, 10,  // 1164, 6
                0, 1, 2, 8, 9, 10,  // 1170, 6
                4, 5, 6, 8, 9, 10,  // 1176, 6
                3, 11,  // 1182, 2
                2, 3, 6, 7, 11,  // 1184, 5
                0, 3, 8, 11,  // 1189, 4
                0, 3, 4, 7, 8, 11,  // 1193, 6
                1, 3, 5, 7, 9, 11,  // 1199, 6
                2, 3, 10, 11,  // 1205, 4
                1, 5, 10, 11,  // 1209, 4
                4, 5, 10, 11,  // 1213, 4
                6, 7, 10, 11,  // 1217, 4
                2, 3, 6, 7, 10, 11,  // 1221, 6
                1, 2, 3, 9, 10, 11,  // 1227, 6
                5, 6, 7, 9, 10, 11,  // 1233, 6
                8, 9, 10, 11,  // 1239, 4
                4, 12,  // 1243, 2
                0, 1, 2, 3, 4, 5, 8, 12,  // 1245, 8
                8, 9, 12,  // 1253, 3
                0, 4, 5, 8, 9, 12,  // 1256, 6
                0, 1, 4, 5, 8, 9, 12,  // 1262, 7
                2, 3, 5, 6, 8, 9, 12,  // 1269, 7
                1, 5, 9, 13,  // 1276, 4
                6, 7, 9, 13,  // 1280, 4
                1, 4, 7, 10, 13,  // 1284, 5
                1, 6, 8, 11, 13,  // 1289, 5
                0, 1, 12, 13,  // 1294, 4
                4, 5, 12, 13,  // 1298, 4
                0, 1, 6, 7, 12, 13,  // 1302, 6
                0, 1, 4, 8, 12, 13,  // 1308, 6
                8, 9, 12, 13,  // 1314, 4
                4, 8, 9, 12, 13,  // 1318, 5
                4, 5, 8, 9, 12, 13,  // 1323, 6
                0, 4, 5, 8, 9, 12, 13,  // 1329, 7
                0, 1, 6, 10, 12, 13,  // 1336, 6
                3, 6, 7, 9, 10, 12, 13,  // 1342, 7
                0, 1, 10, 11, 12, 13,  // 1349, 6
                2, 4, 7, 9, 14,  // 1355, 5
                4, 5, 10, 14,  // 1360, 4
                2, 6, 10, 14,  // 1364, 4
                2, 5, 8, 11, 14,  // 1368, 5
                0, 2, 12, 14,  // 1373, 4
                8, 10, 12, 14,  // 1377, 4
                4, 6, 8, 10, 12, 14,  // 1381, 6
                13, 14,  // 1387, 2
                9, 10, 13, 14,  // 1389, 4
                5, 6, 9, 10, 13, 14,  // 1393, 6
                0, 1, 2, 12, 13, 14,  // 1399, 6
                4, 5, 6, 12, 13, 14,  // 1405, 6
                8, 9, 12, 13, 14,  // 1411, 5
                8, 9, 10, 12, 13, 14,  // 1416, 6
                7, 15,  // 1422, 2
                0, 5, 10, 15,  // 1424, 4
                0, 1, 2, 3, 6, 7, 11, 15,  // 1428, 8
                10, 11, 15,  // 1436, 3
                0, 1, 5, 6, 10, 11, 15,  // 1439, 7
                3, 6, 7, 10, 11, 15,  // 1446, 6
                12, 15,  // 1452, 2
                0, 3, 12, 15,  // 1454, 4
                4, 7, 12, 15,  // 1458, 4
                0, 3, 6, 9, 12, 15,  // 1462, 6
                0, 3, 5, 10, 12, 15,  // 1468, 6
                8, 11, 12, 15,  // 1474, 4
                5, 6, 8, 11, 12, 15,  // 1478, 6
                4, 7, 8, 11, 12, 15,  // 1484, 6
                1, 3, 13, 15,  // 1490, 4
                9, 11, 13, 15,  // 1494, 4
                5, 7, 9, 11, 13, 15,  // 1498, 6
                2, 3, 14, 15,  // 1504, 4
                2, 3, 4, 5, 14, 15,  // 1508, 6
                6, 7, 14, 15,  // 1514, 4
                2, 3, 5, 9, 14, 15,  // 1518, 6
                2, 3, 8, 9, 14, 15,  // 1524, 6
                10, 14, 15,  // 1530, 3
                0, 4, 5, 9, 10, 14, 15,  // 1533, 7
                2, 3, 7, 11, 14, 15,  // 1540, 6
                10, 11, 14, 15,  // 1546, 4
                7, 10, 11, 14, 15,  // 1550, 5
                6, 7, 10, 11, 14, 15,  // 1555, 6
                1, 2, 3, 13, 14, 15,  // 1561, 6
                5, 6, 7, 13, 14, 15,  // 1567, 6
                10, 11, 13, 14, 15,  // 1573, 5
                9, 10, 11, 13, 14, 15,  // 1578, 6
                0, 4, 8, 9, 12, 13, 14, 15,  // 1584, 8
                9, 10, 12, 13, 14, 15,  // 1592, 6
                8, 11, 12, 13, 14, 15,  // 1598, 6
                3, 7, 10, 11, 12, 13, 14, 15,  // 1604, 8
            };
            static const int g_shapeRanges[][2] =
            {
                { 0, 16 },{ 16, 4 },{ 20, 3 },{ 23, 4 },{ 27, 3 },{ 30, 4 },{ 34, 8 },{ 42, 4 },{ 46, 6 },{ 52, 8 },{ 60, 5 },
                { 65, 5 },{ 70, 4 },{ 74, 4 },{ 78, 6 },{ 84, 8 },{ 92, 8 },{ 100, 8 },{ 108, 8 },{ 116, 12 },{ 128, 4 },{ 132, 8 },
                { 140, 8 },{ 148, 10 },{ 158, 6 },{ 164, 8 },{ 172, 12 },{ 184, 8 },{ 192, 5 },{ 197, 3 },{ 200, 4 },{ 204, 6 },{ 210, 8 },
                { 218, 8 },{ 226, 8 },{ 234, 8 },{ 242, 8 },{ 250, 12 },{ 262, 13 },{ 275, 8 },{ 283, 8 },{ 291, 10 },{ 301, 8 },{ 309, 8 },
                { 317, 5 },{ 322, 8 },{ 330, 8 },{ 338, 8 },{ 346, 8 },{ 354, 8 },{ 362, 8 },{ 370, 8 },{ 378, 8 },{ 386, 8 },{ 394, 8 },
                { 402, 8 },{ 410, 8 },{ 418, 4 },{ 422, 8 },{ 430, 6 },{ 436, 8 },{ 444, 10 },{ 454, 8 },{ 462, 12 },{ 474, 8 },{ 482, 8 },
                { 490, 4 },{ 494, 8 },{ 502, 6 },{ 508, 8 },{ 516, 10 },{ 526, 8 },{ 534, 12 },{ 546, 8 },{ 554, 8 },{ 562, 8 },{ 570, 8 },
                { 578, 8 },{ 586, 8 },{ 594, 8 },{ 602, 8 },{ 610, 8 },{ 618, 8 },{ 626, 8 },{ 634, 8 },{ 642, 11 },{ 653, 8 },{ 661, 8 },
                { 669, 6 },{ 675, 8 },{ 683, 8 },{ 691, 3 },{ 694, 4 },{ 698, 8 },{ 706, 8 },{ 714, 8 },{ 722, 8 },{ 730, 8 },{ 738, 10 },
                { 748, 12 },{ 760, 13 },{ 773, 11 },{ 784, 8 },{ 792, 4 },{ 796, 8 },{ 804, 10 },{ 814, 6 },{ 820, 8 },{ 828, 8 },{ 836, 12 },
                { 848, 4 },{ 852, 8 },{ 860, 8 },{ 868, 8 },{ 876, 8 },{ 884, 10 },{ 894, 12 },{ 906, 12 },{ 918, 11 },{ 929, 11 },{ 940, 8 },
                { 948, 10 },{ 958, 12 },{ 970, 8 },{ 978, 12 },{ 990, 13 },{ 1003, 12 },{ 1015, 13 },{ 1028, 12 },{ 1040, 2 },{ 1042, 2 },{ 1044, 4 },
                { 1048, 5 },{ 1053, 3 },{ 1056, 4 },{ 1060, 4 },{ 1064, 6 },{ 1070, 6 },{ 1076, 7 },{ 1083, 4 },{ 1087, 6 },{ 1093, 4 },{ 1097, 4 },
                { 1101, 5 },{ 1106, 6 },{ 1112, 7 },{ 1119, 4 },{ 1123, 2 },{ 1125, 5 },{ 1130, 4 },{ 1134, 4 },{ 1138, 6 },{ 1144, 4 },{ 1148, 4 },
                { 1152, 6 },{ 1158, 6 },{ 1164, 6 },{ 1170, 6 },{ 1176, 6 },{ 1182, 2 },{ 1184, 5 },{ 1189, 4 },{ 1193, 6 },{ 1199, 6 },{ 1205, 4 },
                { 1209, 4 },{ 1213, 4 },{ 1217, 4 },{ 1221, 6 },{ 1227, 6 },{ 1233, 6 },{ 1239, 4 },{ 1243, 2 },{ 1245, 8 },{ 1253, 3 },{ 1256, 6 },
                { 1262, 7 },{ 1269, 7 },{ 1276, 4 },{ 1280, 4 },{ 1284, 5 },{ 1289, 5 },{ 1294, 4 },{ 1298, 4 },{ 1302, 6 },{ 1308, 6 },{ 1314, 4 },
                { 1318, 5 },{ 1323, 6 },{ 1329, 7 },{ 1336, 6 },{ 1342, 7 },{ 1349, 6 },{ 1355, 5 },{ 1360, 4 },{ 1364, 4 },{ 1368, 5 },{ 1373, 4 },
                { 1377, 4 },{ 1381, 6 },{ 1387, 2 },{ 1389, 4 },{ 1393, 6 },{ 1399, 6 },{ 1405, 6 },{ 1411, 5 },{ 1416, 6 },{ 1422, 2 },{ 1424, 4 },
                { 1428, 8 },{ 1436, 3 },{ 1439, 7 },{ 1446, 6 },{ 1452, 2 },{ 1454, 4 },{ 1458, 4 },{ 1462, 6 },{ 1468, 6 },{ 1474, 4 },{ 1478, 6 },
                { 1484, 6 },{ 1490, 4 },{ 1494, 4 },{ 1498, 6 },{ 1504, 4 },{ 1508, 6 },{ 1514, 4 },{ 1518, 6 },{ 1524, 6 },{ 1530, 3 },{ 1533, 7 },
                { 1540, 6 },{ 1546, 4 },{ 1550, 5 },{ 1555, 6 },{ 1561, 6 },{ 1567, 6 },{ 1573, 5 },{ 1578, 6 },{ 1584, 8 },{ 1592, 6 },{ 1598, 6 },
                { 1604, 8 },
            };
            static const int g_shapes1[][2] =
            {
                { 0, 16 }
            };
            static const int g_shapes2[64][2] =
            {
                { 33, 96 },{ 63, 66 },{ 20, 109 },{ 22, 107 },{ 37, 92 },{ 7, 122 },{ 8, 121 },{ 23, 106 },
                { 38, 91 },{ 2, 127 },{ 9, 120 },{ 26, 103 },{ 3, 126 },{ 6, 123 },{ 1, 128 },{ 19, 110 },
                { 15, 114 },{ 124, 5 },{ 72, 57 },{ 115, 14 },{ 125, 4 },{ 70, 59 },{ 100, 29 },{ 60, 69 },
                { 116, 13 },{ 99, 30 },{ 78, 51 },{ 94, 35 },{ 104, 25 },{ 111, 18 },{ 71, 58 },{ 90, 39 },
                { 45, 84 },{ 16, 113 },{ 82, 47 },{ 95, 34 },{ 87, 42 },{ 83, 46 },{ 53, 76 },{ 48, 81 },
                { 68, 61 },{ 105, 24 },{ 98, 31 },{ 88, 41 },{ 75, 54 },{ 43, 86 },{ 52, 77 },{ 117, 12 },
                { 119, 10 },{ 118, 11 },{ 85, 44 },{ 101, 28 },{ 36, 93 },{ 55, 74 },{ 89, 40 },{ 79, 50 },
                { 56, 73 },{ 49, 80 },{ 64, 65 },{ 27, 102 },{ 32, 97 },{ 112, 17 },{ 67, 62 },{ 21, 108 },
            };
            static const int g_shapes3[64][3] =
            {
                { 148, 160, 240 },{ 132, 212, 205 },{ 136, 233, 187 },{ 175, 237, 143 },{ 6, 186, 232 },{ 33, 142, 232 },{ 131, 123, 142 },{ 131, 96, 186 },
                { 6, 171, 110 },{ 1, 18, 110 },{ 1, 146, 123 },{ 33, 195, 66 },{ 20, 51, 66 },{ 20, 178, 96 },{ 2, 177, 106 },{ 211, 4, 59 },
                { 8, 191, 91 },{ 230, 14, 29 },{ 1, 188, 234 },{ 151, 110, 168 },{ 20, 144, 238 },{ 137, 66, 206 },{ 173, 179, 232 },{ 209, 194, 186 },
                { 239, 165, 142 },{ 131, 152, 242 },{ 214, 54, 12 },{ 140, 219, 201 },{ 190, 150, 231 },{ 156, 135, 241 },{ 185, 227, 167 },{ 145, 210, 59 },
                { 138, 174, 106 },{ 189, 229, 14 },{ 176, 133, 106 },{ 78, 178, 195 },{ 111, 146, 171 },{ 216, 180, 196 },{ 217, 181, 193 },{ 184, 228, 166 },
                { 192, 225, 153 },{ 134, 141, 123 },{ 6, 222, 198 },{ 149, 183, 96 },{ 33, 226, 164 },{ 161, 215, 51 },{ 197, 221, 18 },{ 1, 223, 199 },
                { 154, 163, 110 },{ 20, 236, 169 },{ 157, 204, 66 },{ 1, 202, 220 },{ 20, 170, 235 },{ 203, 158, 66 },{ 162, 155, 110 },{ 6, 201, 218 },
                { 139, 135, 123 },{ 33, 167, 224 },{ 182, 150, 96 },{ 19, 200, 213 },{ 63, 207, 159 },{ 147, 172, 109 },{ 129, 130, 128 },{ 208, 14, 59 },
            };

            static const int g_shapeList1[] =
            {
                0,
            };

            static const int g_shapeList2[] =
            {
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
                67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
                78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88,
                89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
                122, 123, 124, 125, 126, 127, 128,
            };

            static const int g_shapeList12[] =
            {
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
                66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
                77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,
                88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98,
                99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
                121, 122, 123, 124, 125, 126, 127, 128,
            };

            static const int g_shapeList3[] =
            {
                1, 2, 4, 6, 8, 12, 14, 18, 19, 20, 29,
                33, 51, 54, 59, 63, 66, 78, 91, 96, 106, 109,
                110, 111, 123, 128, 129, 130, 131, 132, 133, 134, 135,
                136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146,
                147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157,
                158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168,
                169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179,
                180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190,
                191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201,
                202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212,
                213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
                224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234,
                235, 236, 237, 238, 239, 240, 241, 242,
            };

            static const int g_shapeList3Short[] =
            {
                1, 2, 4, 6, 18, 20, 33, 51, 59, 66, 96,
                106, 110, 123, 131, 132, 136, 142, 143, 146, 148, 160,
                171, 175, 177, 178, 186, 187, 195, 205, 211, 212, 232,
                233, 237, 240,
            };

            static const int g_shapeListAll[] =
            {
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
                66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
                77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,
                88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98,
                99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
                121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131,
                132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
                143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153,
                154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164,
                165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
                176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186,
                187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197,
                198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208,
                209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219,
                220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230,
                231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241,
                242,
            };

            static const int g_numShapes1 = sizeof(g_shapeList1) / sizeof(g_shapeList1[0]);
            static const int g_numShapes2 = sizeof(g_shapeList2) / sizeof(g_shapeList2[0]);
            static const int g_numShapes12 = sizeof(g_shapeList12) / sizeof(g_shapeList12[0]);
            static const int g_numShapes3 = sizeof(g_shapeList3) / sizeof(g_shapeList3[0]);
            static const int g_numShapes3Short = sizeof(g_shapeList3Short) / sizeof(g_shapeList3Short[0]);
            static const int g_numShapesAll = sizeof(g_shapeListAll) / sizeof(g_shapeListAll[0]);
            static const int g_numFragments = sizeof(g_fragments) / sizeof(g_fragments[0]);
        }

        struct PackingVector
        {
            uint32_t m_vector[4];
            int m_offset;

            void Init()
            {
                for (int i = 0; i < 4; i++)
                    m_vector[i] = 0;

                m_offset = 0;
            }

            void InitPacked(const uint32_t *v, int bits)
            {
                for (int b = 0; b < bits; b += 32)
                    m_vector[b / 32] = v[b / 32];

                m_offset = bits;
            }

            inline void Pack(ParallelMath::ScalarUInt16 value, int bits)
            {
                int vOffset = m_offset >> 5;
                int bitOffset = m_offset & 0x1f;

                m_vector[vOffset] |= (static_cast<uint32_t>(value) << bitOffset) & static_cast<uint32_t>(0xffffffff);

                int overflowBits = bitOffset + bits - 32;
                if (overflowBits > 0)
                    m_vector[vOffset + 1] |= (static_cast<uint32_t>(value) >> (bits - overflowBits));

                m_offset += bits;
            }

            inline void Flush(uint8_t* output)
            {
                assert(m_offset == 128);

                for (int v = 0; v < 4; v++)
                {
                    uint32_t chunk = m_vector[v];
                    for (int b = 0; b < 4; b++)
                        output[v * 4 + b] = static_cast<uint8_t>((chunk >> (b * 8)) & 0xff);
                }
            }
        };


        struct UnpackingVector
        {
            uint32_t m_vector[4];

            void Init(const uint8_t *bytes)
            {
                for (int i = 0; i < 4; i++)
                    m_vector[i] = 0;

                for (int b = 0; b < 16; b++)
                    m_vector[b / 4] |= (bytes[b] << ((b % 4) * 8));
            }

            inline void UnpackStart(uint32_t *v, int bits)
            {
                for (int b = 0; b < bits; b += 32)
                    v[b / 32] = m_vector[b / 32];

                int entriesShifted = bits / 32;
                int carry = bits % 32;

                for (int i = entriesShifted; i < 4; i++)
                    m_vector[i - entriesShifted] = m_vector[i];

                int entriesRemaining = 4 - entriesShifted;
                if (carry)
                {
                    uint32_t bitMask = (1 << carry) - 1;
                    for (int i = 0; i < 4; i++)
                    {
                        m_vector[i] >>= carry;
                        if (i != 3)
                            m_vector[i] |= (m_vector[i + 1] & bitMask) << (32 - carry);
                    }
                }
            }

            inline ParallelMath::ScalarUInt16 Unpack(int bits)
            {
                uint32_t bitMask = (1 << bits) - 1;

                ParallelMath::ScalarUInt16 result = static_cast<ParallelMath::ScalarUInt16>(m_vector[0] & bitMask);

                for (int i = 0; i < 4; i++)
                {
                    m_vector[i] >>= bits;
                    if (i != 3)
                        m_vector[i] |= (m_vector[i + 1] & bitMask) << (32 - bits);
                }

                return result;
            }
        };

        ParallelMath::Float ScaleHDRValue(const ParallelMath::Float &v, bool isSigned)
        {
            if (isSigned)
            {
                ParallelMath::Float offset = ParallelMath::Select(ParallelMath::Less(v, ParallelMath::MakeFloatZero()), ParallelMath::MakeFloat(-30.0f), ParallelMath::MakeFloat(30.0f));
                return (v * 32.0f + offset) / 31.0f;
            }
            else
                return (v * 64.0f + 30.0f) / 31.0f;
        }

        ParallelMath::SInt16 UnscaleHDRValueSigned(const ParallelMath::SInt16 &v)
        {
#ifdef CVTT_ENABLE_ASSERTS
            for (int i = 0; i < ParallelMath::ParallelSize; i++)
                assert(ParallelMath::Extract(v, i) != -32768)
#endif

                ParallelMath::Int16CompFlag negative = ParallelMath::Less(v, ParallelMath::MakeSInt16(0));
            ParallelMath::UInt15 absComp = ParallelMath::LosslessCast<ParallelMath::UInt15>::Cast(ParallelMath::Select(negative, ParallelMath::SInt16(ParallelMath::MakeSInt16(0) - v), v));

            ParallelMath::UInt31 multiplied = ParallelMath::XMultiply(absComp, ParallelMath::MakeUInt15(31));
            ParallelMath::UInt31 shifted = ParallelMath::RightShift(multiplied, 5);
            ParallelMath::UInt15 absCompScaled = ParallelMath::ToUInt15(shifted);
            ParallelMath::SInt16 signBits = ParallelMath::SelectOrZero(negative, ParallelMath::MakeSInt16(-32768));

            return ParallelMath::LosslessCast<ParallelMath::SInt16>::Cast(absCompScaled) | signBits;
        }

        ParallelMath::UInt15 UnscaleHDRValueUnsigned(const ParallelMath::UInt16 &v)
        {
            return ParallelMath::ToUInt15(ParallelMath::RightShift(ParallelMath::XMultiply(v, ParallelMath::MakeUInt15(31)), 6));
        }

        void UnscaleHDREndpoints(const ParallelMath::AInt16 inEP[2][3], ParallelMath::AInt16 outEP[2][3], bool isSigned)
        {
            for (int epi = 0; epi < 2; epi++)
            {
                for (int ch = 0; ch < 3; ch++)
                {
                    if (isSigned)
                        outEP[epi][ch] = ParallelMath::LosslessCast<ParallelMath::AInt16>::Cast(UnscaleHDRValueSigned(ParallelMath::LosslessCast<ParallelMath::SInt16>::Cast(inEP[epi][ch])));
                    else
                        outEP[epi][ch] = ParallelMath::LosslessCast<ParallelMath::AInt16>::Cast(UnscaleHDRValueUnsigned(ParallelMath::LosslessCast<ParallelMath::UInt16>::Cast(inEP[epi][ch])));
                }
            }
        }

        struct SinglePlaneTemporaries
        {
            UnfinishedEndpoints<3> unfinishedRGB[BC7Data::g_numShapesAll];
            UnfinishedEndpoints<4> unfinishedRGBA[BC7Data::g_numShapes12];

            ParallelMath::UInt15 fragmentBestIndexes[BC7Data::g_numFragments];
            ParallelMath::UInt15 shapeBestEP[BC7Data::g_numShapesAll][2][4];
            ParallelMath::Float shapeBestError[BC7Data::g_numShapesAll];
        };
    }
}

void cvtt::Internal::BC7Computer::TweakAlpha(const MUInt15 original[2], int tweak, int range, MUInt15 result[2])
{
    ParallelMath::RoundTowardNearestForScope roundingMode;

    float tf[2];
    Util::ComputeTweakFactors(tweak, range, tf);

    MFloat base = ParallelMath::ToFloat(original[0]);
    MFloat offs = ParallelMath::ToFloat(original[1]) - base;

    result[0] = ParallelMath::RoundAndConvertToU15(ParallelMath::Clamp(base + offs * tf[0], 0.0f, 255.0f), &roundingMode);
    result[1] = ParallelMath::RoundAndConvertToU15(ParallelMath::Clamp(base + offs * tf[1], 0.0f, 255.0f), &roundingMode);
}

void cvtt::Internal::BC7Computer::Quantize(MUInt15* color, int bits, int channels)
{
    for (int ch = 0; ch < channels; ch++)
        color[ch] = ParallelMath::RightShift(((color[ch] << bits) - color[ch]) + ParallelMath::MakeUInt15(127 + (1 << (7 - bits))), 8);
}

void cvtt::Internal::BC7Computer::QuantizeP(MUInt15* color, int bits, uint16_t p, int channels)
{
    int16_t addend;
    if (p)
        addend = ((1 << (8 - bits)) - 1);
    else
        addend = 255;

    for (int ch = 0; ch < channels; ch++)
    {
        MUInt16 ch16 = ParallelMath::LosslessCast<MUInt16>::Cast(color[ch]);
        ch16 = ParallelMath::RightShift((ch16 << (bits + 1)) - ch16 + addend, 9);
        ch16 = (ch16 << 1) | ParallelMath::MakeUInt16(p);
        color[ch] = ParallelMath::LosslessCast<MUInt15>::Cast(ch16);
    }
}

void cvtt::Internal::BC7Computer::Unquantize(MUInt15* color, int bits, int channels)
{
    for (int ch = 0; ch < channels; ch++)
    {
        MUInt15 clr = color[ch];
        clr = clr << (8 - bits);
        color[ch] = clr | ParallelMath::RightShift(clr, bits);
    }
}

void cvtt::Internal::BC7Computer::CompressEndpoints0(MUInt15 ep[2][4], uint16_t p[2])
{
    for (int j = 0; j < 2; j++)
    {
        QuantizeP(ep[j], 4, p[j], 3);
        Unquantize(ep[j], 5, 3);
        ep[j][3] = ParallelMath::MakeUInt15(255);
    }
}

void cvtt::Internal::BC7Computer::CompressEndpoints1(MUInt15 ep[2][4], uint16_t p)
{
    for (int j = 0; j < 2; j++)
    {
        QuantizeP(ep[j], 6, p, 3);
        Unquantize(ep[j], 7, 3);
        ep[j][3] = ParallelMath::MakeUInt15(255);
    }
}

void cvtt::Internal::BC7Computer::CompressEndpoints2(MUInt15 ep[2][4])
{
    for (int j = 0; j < 2; j++)
    {
        Quantize(ep[j], 5, 3);
        Unquantize(ep[j], 5, 3);
        ep[j][3] = ParallelMath::MakeUInt15(255);
    }
}

void cvtt::Internal::BC7Computer::CompressEndpoints3(MUInt15 ep[2][4], uint16_t p[2])
{
    for (int j = 0; j < 2; j++)
    {
        QuantizeP(ep[j], 7, p[j], 3);
        ep[j][3] = ParallelMath::MakeUInt15(255);
    }
}

void cvtt::Internal::BC7Computer::CompressEndpoints4(MUInt15 epRGB[2][3], MUInt15 epA[2])
{
    for (int j = 0; j < 2; j++)
    {
        Quantize(epRGB[j], 5, 3);
        Unquantize(epRGB[j], 5, 3);

        Quantize(epA + j, 6, 1);
        Unquantize(epA + j, 6, 1);
    }
}

void cvtt::Internal::BC7Computer::CompressEndpoints5(MUInt15 epRGB[2][3], MUInt15 epA[2])
{
    for (int j = 0; j < 2; j++)
    {
        Quantize(epRGB[j], 7, 3);
        Unquantize(epRGB[j], 7, 3);
    }

    // Alpha is full precision
    (void)epA;
}

void cvtt::Internal::BC7Computer::CompressEndpoints6(MUInt15 ep[2][4], uint16_t p[2])
{
    for (int j = 0; j < 2; j++)
        QuantizeP(ep[j], 7, p[j], 4);
}

void cvtt::Internal::BC7Computer::CompressEndpoints7(MUInt15 ep[2][4], uint16_t p[2])
{
    for (int j = 0; j < 2; j++)
    {
        QuantizeP(ep[j], 5, p[j], 4);
        Unquantize(ep[j], 6, 4);
    }
}

void cvtt::Internal::BC7Computer::TrySingleColorRGBAMultiTable(uint32_t flags, const MUInt15 pixels[16][4], const MFloat average[4], int numRealChannels, const uint8_t *fragmentStart, int shapeLength, const MFloat &staticAlphaError, const ParallelMath::Int16CompFlag punchThroughInvalid[4], MFloat& shapeBestError, MUInt15 shapeBestEP[2][4], MUInt15 *fragmentBestIndexes, const float *channelWeightsSq, const cvtt::Tables::BC7SC::Table*const* tables, int numTables, const ParallelMath::RoundTowardNearestForScope *rtn)
{
    MFloat bestAverageError = ParallelMath::MakeFloat(FLT_MAX);

    MUInt15 intAverage[4];
    for (int ch = 0; ch < 4; ch++)
        intAverage[ch] = ParallelMath::RoundAndConvertToU15(average[ch], rtn);

    MUInt15 eps[2][4];
    MUInt15 reconstructed[4];
    MUInt15 index = ParallelMath::MakeUInt15(0);

    for (int epi = 0; epi < 2; epi++)
    {
        for (int ch = 0; ch < 3; ch++)
            eps[epi][ch] = ParallelMath::MakeUInt15(0);
        eps[epi][3] = ParallelMath::MakeUInt15(255);
    }

    for (int ch = 0; ch < 3; ch++)
        reconstructed[ch] = ParallelMath::MakeUInt15(0);
    reconstructed[3] = ParallelMath::MakeUInt15(255);

    // Depending on the target index and parity bits, there are multiple valid solid colors.
    // We want to find the one closest to the actual average.
    MFloat epsAverageDiff = ParallelMath::MakeFloat(FLT_MAX);
    for (int t = 0; t < numTables; t++)
    {
        const cvtt::Tables::BC7SC::Table& table = *(tables[t]);

        ParallelMath::Int16CompFlag pti = punchThroughInvalid[table.m_pBits];

        MUInt15 candidateReconstructed[4];
        MUInt15 candidateEPs[2][4];

        for (int i = 0; i < ParallelMath::ParallelSize; i++)
        {
            for (int ch = 0; ch < numRealChannels; ch++)
            {
                ParallelMath::ScalarUInt16 avgValue = ParallelMath::Extract(intAverage[ch], i);
                assert(avgValue >= 0 && avgValue <= 255);

                const cvtt::Tables::BC7SC::TableEntry &entry = table.m_entries[avgValue];

                ParallelMath::PutUInt15(candidateEPs[0][ch], i, entry.m_min);
                ParallelMath::PutUInt15(candidateEPs[1][ch], i, entry.m_max);
                ParallelMath::PutUInt15(candidateReconstructed[ch], i, entry.m_actualColor);
            }
        }

        MFloat avgError = ParallelMath::MakeFloatZero();
        for (int ch = 0; ch < numRealChannels; ch++)
        {
            MFloat delta = ParallelMath::ToFloat(candidateReconstructed[ch]) - average[ch];
            avgError = avgError + delta * delta * channelWeightsSq[ch];
        }

        ParallelMath::Int16CompFlag better = ParallelMath::FloatFlagToInt16(ParallelMath::Less(avgError, bestAverageError));
        better = ParallelMath::AndNot(pti, better); // Mask out punch-through invalidations

        if (ParallelMath::AnySet(better))
        {
            ParallelMath::ConditionalSet(bestAverageError, ParallelMath::Int16FlagToFloat(better), avgError);

            MUInt15 candidateIndex = ParallelMath::MakeUInt15(table.m_index);

            ParallelMath::ConditionalSet(index, better, candidateIndex);

            for (int ch = 0; ch < numRealChannels; ch++)
                ParallelMath::ConditionalSet(reconstructed[ch], better, candidateReconstructed[ch]);

            for (int epi = 0; epi < 2; epi++)
                for (int ch = 0; ch < numRealChannels; ch++)
                    ParallelMath::ConditionalSet(eps[epi][ch], better, candidateEPs[epi][ch]);
        }
    }

    AggregatedError<4> aggError;
    for (int pxi = 0; pxi < shapeLength; pxi++)
    {
        int px = fragmentStart[pxi];

        BCCommon::ComputeErrorLDR<4>(flags, reconstructed, pixels[px], numRealChannels, aggError);
    }

    MFloat error = aggError.Finalize(flags, channelWeightsSq) + staticAlphaError;

    ParallelMath::Int16CompFlag better = ParallelMath::FloatFlagToInt16(ParallelMath::Less(error, shapeBestError));
    if (ParallelMath::AnySet(better))
    {
        shapeBestError = ParallelMath::Min(shapeBestError, error);
        for (int epi = 0; epi < 2; epi++)
        {
            for (int ch = 0; ch < numRealChannels; ch++)
                ParallelMath::ConditionalSet(shapeBestEP[epi][ch], better, eps[epi][ch]);
        }

        for (int pxi = 0; pxi < shapeLength; pxi++)
            ParallelMath::ConditionalSet(fragmentBestIndexes[pxi], better, index);
    }
}

void cvtt::Internal::BC7Computer::TrySinglePlane(uint32_t flags, const MUInt15 pixels[16][4], const MFloat floatPixels[16][4], const float channelWeights[4], const BC7EncodingPlan &encodingPlan, int numRefineRounds, BC67::WorkInfo& work, const ParallelMath::RoundTowardNearestForScope *rtn)
{
    if (numRefineRounds < 1)
        numRefineRounds = 1;

    float channelWeightsSq[4];

    for (int ch = 0; ch < 4; ch++)
        channelWeightsSq[ch] = channelWeights[ch] * channelWeights[ch];

    SinglePlaneTemporaries temps;

    MUInt15 maxAlpha = ParallelMath::MakeUInt15(0);
    MUInt15 minAlpha = ParallelMath::MakeUInt15(255);
    ParallelMath::Int16CompFlag isPunchThrough = ParallelMath::MakeBoolInt16(true);
    for (int px = 0; px < 16; px++)
    {
        MUInt15 a = pixels[px][3];
        maxAlpha = ParallelMath::Max(maxAlpha, a);
        minAlpha = ParallelMath::Min(minAlpha, a);

        isPunchThrough = (isPunchThrough & (ParallelMath::Equal(a, ParallelMath::MakeUInt15(0)) | ParallelMath::Equal(a, ParallelMath::MakeUInt15(255))));
    }

    ParallelMath::Int16CompFlag blockHasNonMaxAlpha = ParallelMath::Less(minAlpha, ParallelMath::MakeUInt15(255));
    ParallelMath::Int16CompFlag blockHasNonZeroAlpha = ParallelMath::Less(ParallelMath::MakeUInt15(0), maxAlpha);

    bool anyBlockHasAlpha = ParallelMath::AnySet(blockHasNonMaxAlpha);

    // Try RGB modes if any block has a min alpha 251 or higher
    bool allowRGBModes = ParallelMath::AnySet(ParallelMath::Less(ParallelMath::MakeUInt15(250), minAlpha));

    // Try mode 7 if any block has alpha.
    // Mode 7 is almost never selected for RGB blocks because mode 4 has very accurate 7.7.7.1 endpoints
    // and its parity bit doesn't affect alpha, meaning mode 7 can only be better in extremely specific
    // situations, and only by at most 1 unit of error per pixel.
    bool allowMode7 = anyBlockHasAlpha || (encodingPlan.mode7RGBPartitionEnabled != 0);

    MFloat preWeightedPixels[16][4];

    BCCommon::PreWeightPixelsLDR<4>(preWeightedPixels, pixels, channelWeights);

    // Get initial RGB endpoints
    if (allowRGBModes)
    {
        const uint8_t *shapeList = encodingPlan.rgbShapeList;
        int numShapesToEvaluate = encodingPlan.rgbNumShapesToEvaluate;

        for (int shapeIter = 0; shapeIter < numShapesToEvaluate; shapeIter++)
        {
            int shape = shapeList[shapeIter];

            int shapeStart = BC7Data::g_shapeRanges[shape][0];
            int shapeSize = BC7Data::g_shapeRanges[shape][1];

            EndpointSelector<3, 8> epSelector;

            for (int epPass = 0; epPass < NumEndpointSelectorPasses; epPass++)
            {
                for (int spx = 0; spx < shapeSize; spx++)
                {
                    int px = BC7Data::g_fragments[shapeStart + spx];
                    epSelector.ContributePass(preWeightedPixels[px], epPass, ParallelMath::MakeFloat(1.0f));
                }
                epSelector.FinishPass(epPass);
            }
            temps.unfinishedRGB[shape] = epSelector.GetEndpoints(channelWeights);
        }
    }

    // Get initial RGBA endpoints
    {
        const uint8_t *shapeList = encodingPlan.rgbaShapeList;
        int numShapesToEvaluate = encodingPlan.rgbaNumShapesToEvaluate;

        for (int shapeIter = 0; shapeIter < numShapesToEvaluate; shapeIter++)
        {
            int shape = shapeList[shapeIter];

            if (anyBlockHasAlpha || !allowRGBModes)
            {
                int shapeStart = BC7Data::g_shapeRanges[shape][0];
                int shapeSize = BC7Data::g_shapeRanges[shape][1];

                EndpointSelector<4, 8> epSelector;

                for (int epPass = 0; epPass < NumEndpointSelectorPasses; epPass++)
                {
                    for (int spx = 0; spx < shapeSize; spx++)
                    {
                        int px = BC7Data::g_fragments[shapeStart + spx];
                        epSelector.ContributePass(preWeightedPixels[px], epPass, ParallelMath::MakeFloat(1.0f));
                    }
                    epSelector.FinishPass(epPass);
                }
                temps.unfinishedRGBA[shape] = epSelector.GetEndpoints(channelWeights);
            }
            else
            {
                temps.unfinishedRGBA[shape] = temps.unfinishedRGB[shape].ExpandTo<4>(255);
            }
        }
    }

    for (uint16_t mode = 0; mode <= 7; mode++)
    {
        if (mode == 4 || mode == 5)
            continue;

        if (mode < 4 && !allowRGBModes)
            continue;

        if (mode == 7 && !allowMode7)
            continue;

        uint64_t partitionEnabledBits = 0;
        switch (mode)
        {
        case 0:
            partitionEnabledBits = encodingPlan.mode0PartitionEnabled;
            break;
        case 1:
            partitionEnabledBits = encodingPlan.mode1PartitionEnabled;
            break;
        case 2:
            partitionEnabledBits = encodingPlan.mode2PartitionEnabled;
            break;
        case 3:
            partitionEnabledBits = encodingPlan.mode3PartitionEnabled;
            break;
        case 6:
            partitionEnabledBits = encodingPlan.mode6Enabled ? 1 : 0;
            break;
        case 7:
            if (anyBlockHasAlpha)
                partitionEnabledBits = encodingPlan.mode7RGBAPartitionEnabled;
            else
                partitionEnabledBits = encodingPlan.mode7RGBPartitionEnabled;
            break;
        default:
            break;
        }

        bool isRGB = (mode < 4);

        unsigned int numPartitions = 1 << BC7Data::g_modes[mode].m_partitionBits;
        int numSubsets = BC7Data::g_modes[mode].m_numSubsets;
        int indexPrec = BC7Data::g_modes[mode].m_indexBits;

        int parityBitMax = 1;
        if (BC7Data::g_modes[mode].m_pBitMode == BC7Data::PBitMode_PerEndpoint)
            parityBitMax = 4;
        else if (BC7Data::g_modes[mode].m_pBitMode == BC7Data::PBitMode_PerSubset)
            parityBitMax = 2;

        int numRealChannels = isRGB ? 3 : 4;

        int numShapes;
        const int *shapeList;

        if (numSubsets == 1)
        {
            numShapes = BC7Data::g_numShapes1;
            shapeList = BC7Data::g_shapeList1;
        }
        else if (numSubsets == 2)
        {
            numShapes = BC7Data::g_numShapes2;
            shapeList = BC7Data::g_shapeList2;
        }
        else
        {
            assert(numSubsets == 3);
            if (numPartitions == 16)
            {
                numShapes = BC7Data::g_numShapes3Short;
                shapeList = BC7Data::g_shapeList3Short;
            }
            else
            {
                assert(numPartitions == 64);
                numShapes = BC7Data::g_numShapes3;
                shapeList = BC7Data::g_shapeList3;
            }
        }

        for (int slot = 0; slot < BC7Data::g_numShapesAll; slot++)
            temps.shapeBestError[slot] = ParallelMath::MakeFloat(FLT_MAX);

        for (int shapeIter = 0; shapeIter < numShapes; shapeIter++)
        {
            int shape = shapeList[shapeIter];

            int numTweakRounds = 0;
            if (isRGB)
                numTweakRounds = encodingPlan.seedPointsForShapeRGB[shape];
            else
                numTweakRounds = encodingPlan.seedPointsForShapeRGBA[shape];

            if (numTweakRounds == 0)
                continue;

            if (numTweakRounds > MaxTweakRounds)
                numTweakRounds = MaxTweakRounds;

            int shapeStart = BC7Data::g_shapeRanges[shape][0];
            int shapeLength = BC7Data::g_shapeRanges[shape][1];

            AggregatedError<1> alphaAggError;
            if (isRGB && anyBlockHasAlpha)
            {
                MUInt15 filledAlpha[1] = { ParallelMath::MakeUInt15(255) };

                for (int pxi = 0; pxi < shapeLength; pxi++)
                {
                    int px = BC7Data::g_fragments[shapeStart + pxi];
                    MUInt15 original[1] = { pixels[px][3] };
                    BCCommon::ComputeErrorLDR<1>(flags, filledAlpha, original, alphaAggError);
                }
            }

            float alphaWeightsSq[1] = { channelWeightsSq[3] };
            MFloat staticAlphaError = alphaAggError.Finalize(flags, alphaWeightsSq);

            MUInt15 tweakBaseEP[MaxTweakRounds][2][4];

            for (int tweak = 0; tweak < numTweakRounds; tweak++)
            {
                if (isRGB)
                {
                    temps.unfinishedRGB[shape].FinishLDR(tweak, 1 << indexPrec, tweakBaseEP[tweak][0], tweakBaseEP[tweak][1]);
                    tweakBaseEP[tweak][0][3] = tweakBaseEP[tweak][1][3] = ParallelMath::MakeUInt15(255);
                }
                else
                {
                    temps.unfinishedRGBA[shape].FinishLDR(tweak, 1 << indexPrec, tweakBaseEP[tweak][0], tweakBaseEP[tweak][1]);
                }
            }

            ParallelMath::Int16CompFlag punchThroughInvalid[4];
            for (int pIter = 0; pIter < parityBitMax; pIter++)
            {
                punchThroughInvalid[pIter] = ParallelMath::MakeBoolInt16(false);

                if ((flags & Flags::BC7_RespectPunchThrough) && (mode == 6 || mode == 7))
                {
                    // Modes 6 and 7 have parity bits that affect alpha
                    if (pIter == 0)
                        punchThroughInvalid[pIter] = (isPunchThrough & blockHasNonZeroAlpha);
                    else if (pIter == parityBitMax - 1)
                        punchThroughInvalid[pIter] = (isPunchThrough & blockHasNonMaxAlpha);
                    else
                        punchThroughInvalid[pIter] = isPunchThrough;
                }
            }

            for (int pIter = 0; pIter < parityBitMax; pIter++)
            {
                if (ParallelMath::AllSet(punchThroughInvalid[pIter]))
                    continue;

                bool needPunchThroughCheck = ParallelMath::AnySet(punchThroughInvalid[pIter]);

                for (int tweak = 0; tweak < numTweakRounds; tweak++)
                {
                    uint16_t p[2];
                    p[0] = (pIter & 1);
                    p[1] = ((pIter >> 1) & 1);

                    MUInt15 ep[2][4];

                    for (int epi = 0; epi < 2; epi++)
                        for (int ch = 0; ch < 4; ch++)
                            ep[epi][ch] = tweakBaseEP[tweak][epi][ch];

                    for (int refine = 0; refine < numRefineRounds; refine++)
                    {
                        switch (mode)
                        {
                        case 0:
                            CompressEndpoints0(ep, p);
                            break;
                        case 1:
                            CompressEndpoints1(ep, p[0]);
                            break;
                        case 2:
                            CompressEndpoints2(ep);
                            break;
                        case 3:
                            CompressEndpoints3(ep, p);
                            break;
                        case 6:
                            CompressEndpoints6(ep, p);
                            break;
                        case 7:
                            CompressEndpoints7(ep, p);
                            break;
                        default:
                            assert(false);
                            break;
                        };

                        MFloat shapeError = ParallelMath::MakeFloatZero();

                        IndexSelector<4> indexSelector;
                        indexSelector.Init<false>(channelWeights, ep, 1 << indexPrec);

                        EndpointRefiner<4> epRefiner;
                        epRefiner.Init(1 << indexPrec, channelWeights);

                        MUInt15 indexes[16];

                        AggregatedError<4> aggError;
                        for (int pxi = 0; pxi < shapeLength; pxi++)
                        {
                            int px = BC7Data::g_fragments[shapeStart + pxi];

                            MUInt15 index;
                            MUInt15 reconstructed[4];

                            index = indexSelector.SelectIndexLDR(floatPixels[px], rtn);
                            indexSelector.ReconstructLDR_BC7(index, reconstructed, numRealChannels);

                            if (flags & cvtt::Flags::BC7_FastIndexing)
                                BCCommon::ComputeErrorLDR<4>(flags, reconstructed, pixels[px], numRealChannels, aggError);
                            else
                            {
                                MFloat error = BCCommon::ComputeErrorLDRSimple<4>(flags, reconstructed, pixels[px], numRealChannels, channelWeightsSq);

                                MUInt15 altIndexes[2];
                                altIndexes[0] = ParallelMath::Max(index, ParallelMath::MakeUInt15(1)) - ParallelMath::MakeUInt15(1);
                                altIndexes[1] = ParallelMath::Min(index + ParallelMath::MakeUInt15(1), ParallelMath::MakeUInt15(static_cast<uint16_t>((1 << indexPrec) - 1)));

                                for (int ii = 0; ii < 2; ii++)
                                {
                                    indexSelector.ReconstructLDR_BC7(altIndexes[ii], reconstructed, numRealChannels);

                                    MFloat altError = BCCommon::ComputeErrorLDRSimple<4>(flags, reconstructed, pixels[px], numRealChannels, channelWeightsSq);
                                    ParallelMath::Int16CompFlag better = ParallelMath::FloatFlagToInt16(ParallelMath::Less(altError, error));
                                    error = ParallelMath::Min(error, altError);
                                    ParallelMath::ConditionalSet(index, better, altIndexes[ii]);
                                }

                                shapeError = shapeError + error;
                            }

                            if (refine != numRefineRounds - 1)
                                epRefiner.ContributeUnweightedPW(preWeightedPixels[px], index, numRealChannels);

                            indexes[pxi] = index;
                        }

                        if (flags & cvtt::Flags::BC7_FastIndexing)
                            shapeError = aggError.Finalize(flags, channelWeightsSq);

                        if (isRGB)
                            shapeError = shapeError + staticAlphaError;

                        ParallelMath::FloatCompFlag shapeErrorBetter;
                        ParallelMath::Int16CompFlag shapeErrorBetter16;

                        shapeErrorBetter = ParallelMath::Less(shapeError, temps.shapeBestError[shape]);
                        shapeErrorBetter16 = ParallelMath::FloatFlagToInt16(shapeErrorBetter);

                        if (ParallelMath::AnySet(shapeErrorBetter16))
                        {
                            bool punchThroughOK = true;
                            if (needPunchThroughCheck)
                            {
                                shapeErrorBetter16 = ParallelMath::AndNot(punchThroughInvalid[pIter], shapeErrorBetter16);
                                shapeErrorBetter = ParallelMath::Int16FlagToFloat(shapeErrorBetter16);

                                if (!ParallelMath::AnySet(shapeErrorBetter16))
                                    punchThroughOK = false;
                            }

                            if (punchThroughOK)
                            {
                                ParallelMath::ConditionalSet(temps.shapeBestError[shape], shapeErrorBetter, shapeError);
                                for (int epi = 0; epi < 2; epi++)
                                    for (int ch = 0; ch < numRealChannels; ch++)
                                        ParallelMath::ConditionalSet(temps.shapeBestEP[shape][epi][ch], shapeErrorBetter16, ep[epi][ch]);

                                for (int pxi = 0; pxi < shapeLength; pxi++)
                                    ParallelMath::ConditionalSet(temps.fragmentBestIndexes[shapeStart + pxi], shapeErrorBetter16, indexes[pxi]);
                            }
                        }

                        if (refine != numRefineRounds - 1)
                            epRefiner.GetRefinedEndpointsLDR(ep, numRealChannels, rtn);
                    } // refine
                } // tweak
            } // p

            if (flags & cvtt::Flags::BC7_TrySingleColor)
            {
                MUInt15 total[4];
                for (int ch = 0; ch < 4; ch++)
                    total[ch] = ParallelMath::MakeUInt15(0);

                for (int pxi = 0; pxi < shapeLength; pxi++)
                {
                    int px = BC7Data::g_fragments[shapeStart + pxi];
                    for (int ch = 0; ch < 4; ch++)
                        total[ch] = total[ch] + pixels[pxi][ch];
                }

                MFloat rcpShapeLength = ParallelMath::MakeFloat(1.0f / static_cast<float>(shapeLength));
                MFloat average[4];
                for (int ch = 0; ch < 4; ch++)
                    average[ch] = ParallelMath::ToFloat(total[ch]) * rcpShapeLength;

                const uint8_t *fragment = BC7Data::g_fragments + shapeStart;
                MFloat &shapeBestError = temps.shapeBestError[shape];
                MUInt15 (&shapeBestEP)[2][4] = temps.shapeBestEP[shape];
                MUInt15 *fragmentBestIndexes = temps.fragmentBestIndexes + shapeStart;

                const cvtt::Tables::BC7SC::Table **scTables = NULL;
                int numSCTables = 0;

                const cvtt::Tables::BC7SC::Table *tables0[] =
                {
                    &cvtt::Tables::BC7SC::g_mode0_p00_i1,
                    &cvtt::Tables::BC7SC::g_mode0_p00_i2,
                    &cvtt::Tables::BC7SC::g_mode0_p00_i3,
                    &cvtt::Tables::BC7SC::g_mode0_p01_i1,
                    &cvtt::Tables::BC7SC::g_mode0_p01_i2,
                    &cvtt::Tables::BC7SC::g_mode0_p01_i3,
                    &cvtt::Tables::BC7SC::g_mode0_p10_i1,
                    &cvtt::Tables::BC7SC::g_mode0_p10_i2,
                    &cvtt::Tables::BC7SC::g_mode0_p10_i3,
                    &cvtt::Tables::BC7SC::g_mode0_p11_i1,
                    &cvtt::Tables::BC7SC::g_mode0_p11_i2,
                    &cvtt::Tables::BC7SC::g_mode0_p11_i3,
                };

                const cvtt::Tables::BC7SC::Table *tables1[] =
                {
                    &cvtt::Tables::BC7SC::g_mode1_p0_i1,
                    &cvtt::Tables::BC7SC::g_mode1_p0_i2,
                    &cvtt::Tables::BC7SC::g_mode1_p0_i3,
                    &cvtt::Tables::BC7SC::g_mode1_p1_i1,
                    &cvtt::Tables::BC7SC::g_mode1_p1_i2,
                    &cvtt::Tables::BC7SC::g_mode1_p1_i3,
                };

                const cvtt::Tables::BC7SC::Table *tables2[] =
                {
                    &cvtt::Tables::BC7SC::g_mode2,
                };

                const cvtt::Tables::BC7SC::Table *tables3[] =
                {
                    &cvtt::Tables::BC7SC::g_mode3_p0,
                    &cvtt::Tables::BC7SC::g_mode3_p1,
                };

                const cvtt::Tables::BC7SC::Table *tables6[] =
                {
                    &cvtt::Tables::BC7SC::g_mode6_p0_i1,
                    &cvtt::Tables::BC7SC::g_mode6_p0_i2,
                    &cvtt::Tables::BC7SC::g_mode6_p0_i3,
                    &cvtt::Tables::BC7SC::g_mode6_p0_i4,
                    &cvtt::Tables::BC7SC::g_mode6_p0_i5,
                    &cvtt::Tables::BC7SC::g_mode6_p0_i6,
                    &cvtt::Tables::BC7SC::g_mode6_p0_i7,
                    &cvtt::Tables::BC7SC::g_mode6_p1_i1,
                    &cvtt::Tables::BC7SC::g_mode6_p1_i2,
                    &cvtt::Tables::BC7SC::g_mode6_p1_i3,
                    &cvtt::Tables::BC7SC::g_mode6_p1_i4,
                    &cvtt::Tables::BC7SC::g_mode6_p1_i5,
                    &cvtt::Tables::BC7SC::g_mode6_p1_i6,
                    &cvtt::Tables::BC7SC::g_mode6_p1_i7,
                };

                const cvtt::Tables::BC7SC::Table *tables7[] =
                {
                    &cvtt::Tables::BC7SC::g_mode7_p00,
                    &cvtt::Tables::BC7SC::g_mode7_p01,
                    &cvtt::Tables::BC7SC::g_mode7_p10,
                    &cvtt::Tables::BC7SC::g_mode7_p11,
                };

                switch (mode)
                {
                case 0:
                {
                    scTables = tables0;
                    numSCTables = sizeof(tables0) / sizeof(tables0[0]);
                }
                break;
                case 1:
                {
                    scTables = tables1;
                    numSCTables = sizeof(tables1) / sizeof(tables1[0]);
                }
                break;
                case 2:
                {

                    scTables = tables2;
                    numSCTables = sizeof(tables2) / sizeof(tables2[0]);
                }
                break;
                case 3:
                {
                    scTables = tables3;
                    numSCTables = sizeof(tables3) / sizeof(tables3[0]);
                }
                break;
                case 6:
                {
                    scTables = tables6;
                    numSCTables = sizeof(tables6) / sizeof(tables6[0]);
                }
                break;
                case 7:
                {
                    scTables = tables7;
                    numSCTables = sizeof(tables7) / sizeof(tables7[0]);
                }
                break;
                default:
                    assert(false);
                    break;
                }

                TrySingleColorRGBAMultiTable(flags, pixels, average, numRealChannels, fragment, shapeLength, staticAlphaError, punchThroughInvalid, shapeBestError, shapeBestEP, fragmentBestIndexes, channelWeightsSq, scTables, numSCTables, rtn);
            }
        } // shapeIter

        uint64_t partitionsEnabledBits = 0xffffffffffffffffULL;

        switch (mode)
        {
        case 0:
            partitionsEnabledBits = encodingPlan.mode0PartitionEnabled;
            break;
        case 1:
            partitionsEnabledBits = encodingPlan.mode1PartitionEnabled;
            break;
        case 2:
            partitionsEnabledBits = encodingPlan.mode2PartitionEnabled;
            break;
        case 3:
            partitionsEnabledBits = encodingPlan.mode3PartitionEnabled;
            break;
        case 6:
            partitionsEnabledBits = encodingPlan.mode6Enabled ? 1 : 0;
            break;
        case 7:
            if (anyBlockHasAlpha)
                partitionEnabledBits = encodingPlan.mode7RGBAPartitionEnabled;
            else
                partitionEnabledBits = encodingPlan.mode7RGBPartitionEnabled;
            break;
        default:
            break;
        };

        for (uint16_t partition = 0; partition < numPartitions; partition++)
        {
            if (((partitionsEnabledBits >> partition) & 1) == 0)
                continue;

            const int *partitionShapes;
            if (numSubsets == 1)
                partitionShapes = BC7Data::g_shapes1[partition];
            else if (numSubsets == 2)
                partitionShapes = BC7Data::g_shapes2[partition];
            else
            {
                assert(numSubsets == 3);
                partitionShapes = BC7Data::g_shapes3[partition];
            }

            MFloat totalError = ParallelMath::MakeFloatZero();
            for (int subset = 0; subset < numSubsets; subset++)
                totalError = totalError + temps.shapeBestError[partitionShapes[subset]];

            ParallelMath::FloatCompFlag errorBetter = ParallelMath::Less(totalError, work.m_error);
            ParallelMath::Int16CompFlag errorBetter16 = ParallelMath::FloatFlagToInt16(errorBetter);

            if (mode == 7 && anyBlockHasAlpha)
            {
                // Some lanes could be better, but we filter them out to ensure consistency with scalar
                bool isRGBAllowedForThisPartition = (((encodingPlan.mode7RGBPartitionEnabled >> partition) & 1) != 0);

                if (!isRGBAllowedForThisPartition)
                {
                    errorBetter16 = (errorBetter16 & blockHasNonMaxAlpha);
                    errorBetter = ParallelMath::Int16FlagToFloat(errorBetter16);
                }
            }

            if (ParallelMath::AnySet(errorBetter16))
            {
                for (int subset = 0; subset < numSubsets; subset++)
                {
                    int shape = partitionShapes[subset];
                    int shapeStart = BC7Data::g_shapeRanges[shape][0];
                    int shapeLength = BC7Data::g_shapeRanges[shape][1];

                    for (int epi = 0; epi < 2; epi++)
                        for (int ch = 0; ch < 4; ch++)
                            ParallelMath::ConditionalSet(work.m_ep[subset][epi][ch], errorBetter16, temps.shapeBestEP[shape][epi][ch]);

                    for (int pxi = 0; pxi < shapeLength; pxi++)
                    {
                        int px = BC7Data::g_fragments[shapeStart + pxi];
                        ParallelMath::ConditionalSet(work.m_indexes[px], errorBetter16, temps.fragmentBestIndexes[shapeStart + pxi]);
                    }
                }

                ParallelMath::ConditionalSet(work.m_error, errorBetter, totalError);
                ParallelMath::ConditionalSet(work.m_mode, errorBetter16, ParallelMath::MakeUInt15(mode));
                ParallelMath::ConditionalSet(work.m_u.m_partition, errorBetter16, ParallelMath::MakeUInt15(partition));
            }
        }
    }
}

void cvtt::Internal::BC7Computer::TryDualPlane(uint32_t flags, const MUInt15 pixels[16][4], const MFloat floatPixels[16][4], const float channelWeights[4], const BC7EncodingPlan &encodingPlan, int numRefineRounds, BC67::WorkInfo& work, const ParallelMath::RoundTowardNearestForScope *rtn)
{
    // TODO: These error calculations are not optimal for weight-by-alpha, but this routine needs to be mostly rewritten for that.
    // The alpha/color solutions are co-dependent in that case, but a good way to solve it would probably be to
    // solve the alpha channel first, then solve the RGB channels, which in turn breaks down into two cases:
    // - Separate alpha channel, then weighted RGB
    // - Alpha+2 other channels, then the independent channel
    if (numRefineRounds < 1)
        numRefineRounds = 1;

    float channelWeightsSq[4];
    for (int ch = 0; ch < 4; ch++)
        channelWeightsSq[ch] = channelWeights[ch] * channelWeights[ch];

    for (uint16_t mode = 4; mode <= 5; mode++)
    {
        int numSP[2] = { 0, 0 };

        for (uint16_t rotation = 0; rotation < 4; rotation++)
        {
            if (mode == 4)
            {
                numSP[0] = encodingPlan.mode4SP[rotation][0];
                numSP[1] = encodingPlan.mode4SP[rotation][1];
            }
            else
                numSP[0] = numSP[1] = encodingPlan.mode5SP[rotation];

            if (numSP[0] == 0 && numSP[1] == 0)
                continue;

            int alphaChannel = (rotation + 3) & 3;
            int redChannel = (rotation == 1) ? 3 : 0;
            int greenChannel = (rotation == 2) ? 3 : 1;
            int blueChannel = (rotation == 3) ? 3 : 2;

            MUInt15 rotatedRGB[16][3];
            MFloat floatRotatedRGB[16][3];

            for (int px = 0; px < 16; px++)
            {
                rotatedRGB[px][0] = pixels[px][redChannel];
                rotatedRGB[px][1] = pixels[px][greenChannel];
                rotatedRGB[px][2] = pixels[px][blueChannel];

                for (int ch = 0; ch < 3; ch++)
                    floatRotatedRGB[px][ch] = ParallelMath::ToFloat(rotatedRGB[px][ch]);
            }

            uint16_t maxIndexSelector = (mode == 4) ? 2 : 1;

            float rotatedRGBWeights[3] = { channelWeights[redChannel], channelWeights[greenChannel], channelWeights[blueChannel] };
            float rotatedRGBWeightsSq[3] = { channelWeightsSq[redChannel], channelWeightsSq[greenChannel], channelWeightsSq[blueChannel] };
            float rotatedAlphaWeight[1] = { channelWeights[alphaChannel] };
            float rotatedAlphaWeightSq[1] = { channelWeightsSq[alphaChannel] };

            float uniformWeight[1] = { 1.0f };   // Since the alpha channel is independent, there's no need to bother with weights when doing refinement or selection, only error

            MFloat preWeightedRotatedRGB[16][3];
            BCCommon::PreWeightPixelsLDR<3>(preWeightedRotatedRGB, rotatedRGB, rotatedRGBWeights);

            for (uint16_t indexSelector = 0; indexSelector < maxIndexSelector; indexSelector++)
            {
                int numTweakRounds = numSP[indexSelector];

                if (numTweakRounds <= 0)
                    continue;

                if (numTweakRounds > MaxTweakRounds)
                    numTweakRounds = MaxTweakRounds;

                EndpointSelector<3, 8> rgbSelector;

                for (int epPass = 0; epPass < NumEndpointSelectorPasses; epPass++)
                {
                    for (int px = 0; px < 16; px++)
                        rgbSelector.ContributePass(preWeightedRotatedRGB[px], epPass, ParallelMath::MakeFloat(1.0f));

                    rgbSelector.FinishPass(epPass);
                }

                MUInt15 alphaRange[2];

                alphaRange[0] = alphaRange[1] = pixels[0][alphaChannel];
                for (int px = 1; px < 16; px++)
                {
                    alphaRange[0] = ParallelMath::Min(pixels[px][alphaChannel], alphaRange[0]);
                    alphaRange[1] = ParallelMath::Max(pixels[px][alphaChannel], alphaRange[1]);
                }

                int rgbPrec = 0;
                int alphaPrec = 0;

                if (mode == 4)
                {
                    rgbPrec = indexSelector ? 3 : 2;
                    alphaPrec = indexSelector ? 2 : 3;
                }
                else
                    rgbPrec = alphaPrec = 2;

                UnfinishedEndpoints<3> unfinishedRGB = rgbSelector.GetEndpoints(rotatedRGBWeights);

                MFloat bestRGBError = ParallelMath::MakeFloat(FLT_MAX);
                MFloat bestAlphaError = ParallelMath::MakeFloat(FLT_MAX);

                MUInt15 bestRGBIndexes[16];
                MUInt15 bestAlphaIndexes[16];
                MUInt15 bestEP[2][4];

                for (int px = 0; px < 16; px++)
                    bestRGBIndexes[px] = bestAlphaIndexes[px] = ParallelMath::MakeUInt15(0);

                for (int tweak = 0; tweak < numTweakRounds; tweak++)
                {
                    MUInt15 rgbEP[2][3];
                    MUInt15 alphaEP[2];

                    unfinishedRGB.FinishLDR(tweak, 1 << rgbPrec, rgbEP[0], rgbEP[1]);

                    TweakAlpha(alphaRange, tweak, 1 << alphaPrec, alphaEP);

                    for (int refine = 0; refine < numRefineRounds; refine++)
                    {
                        if (mode == 4)
                            CompressEndpoints4(rgbEP, alphaEP);
                        else
                            CompressEndpoints5(rgbEP, alphaEP);


                        IndexSelector<1> alphaIndexSelector;
                        IndexSelector<3> rgbIndexSelector;

                        {
                            MUInt15 alphaEPTemp[2][1] = { { alphaEP[0] },{ alphaEP[1] } };
                            alphaIndexSelector.Init<false>(uniformWeight, alphaEPTemp, 1 << alphaPrec);
                        }
                        rgbIndexSelector.Init<false>(rotatedRGBWeights, rgbEP, 1 << rgbPrec);

                        EndpointRefiner<3> rgbRefiner;
                        EndpointRefiner<1> alphaRefiner;

                        rgbRefiner.Init(1 << rgbPrec, rotatedRGBWeights);
                        alphaRefiner.Init(1 << alphaPrec, uniformWeight);

                        MFloat errorRGB = ParallelMath::MakeFloatZero();
                        MFloat errorA = ParallelMath::MakeFloatZero();

                        MUInt15 rgbIndexes[16];
                        MUInt15 alphaIndexes[16];

                        AggregatedError<3> rgbAggError;
                        AggregatedError<1> alphaAggError;

                        for (int px = 0; px < 16; px++)
                        {
                            MUInt15 rgbIndex = rgbIndexSelector.SelectIndexLDR(floatRotatedRGB[px], rtn);
                            MUInt15 alphaIndex = alphaIndexSelector.SelectIndexLDR(floatPixels[px] + alphaChannel, rtn);

                            MUInt15 reconstructedRGB[3];
                            MUInt15 reconstructedAlpha[1];

                            rgbIndexSelector.ReconstructLDR_BC7(rgbIndex, reconstructedRGB);
                            alphaIndexSelector.ReconstructLDR_BC7(alphaIndex, reconstructedAlpha);

                            if (flags & cvtt::Flags::BC7_FastIndexing)
                            {
                                BCCommon::ComputeErrorLDR<3>(flags, reconstructedRGB, rotatedRGB[px], rgbAggError);
                                BCCommon::ComputeErrorLDR<1>(flags, reconstructedAlpha, pixels[px] + alphaChannel, alphaAggError);
                            }
                            else
                            {
                                AggregatedError<3> baseRGBAggError;
                                AggregatedError<1> baseAlphaAggError;

                                BCCommon::ComputeErrorLDR<3>(flags, reconstructedRGB, rotatedRGB[px], baseRGBAggError);
                                BCCommon::ComputeErrorLDR<1>(flags, reconstructedAlpha, pixels[px] + alphaChannel, baseAlphaAggError);

                                MFloat rgbError = baseRGBAggError.Finalize(flags, rotatedRGBWeightsSq);
                                MFloat alphaError = baseAlphaAggError.Finalize(flags, rotatedAlphaWeightSq);

                                MUInt15 altRGBIndexes[2];
                                MUInt15 altAlphaIndexes[2];

                                altRGBIndexes[0] = ParallelMath::Max(rgbIndex, ParallelMath::MakeUInt15(1)) - ParallelMath::MakeUInt15(1);
                                altRGBIndexes[1] = ParallelMath::Min(rgbIndex + ParallelMath::MakeUInt15(1), ParallelMath::MakeUInt15(static_cast<uint16_t>((1 << rgbPrec) - 1)));

                                altAlphaIndexes[0] = ParallelMath::Max(alphaIndex, ParallelMath::MakeUInt15(1)) - ParallelMath::MakeUInt15(1);
                                altAlphaIndexes[1] = ParallelMath::Min(alphaIndex + ParallelMath::MakeUInt15(1), ParallelMath::MakeUInt15(static_cast<uint16_t>((1 << alphaPrec) - 1)));

                                for (int ii = 0; ii < 2; ii++)
                                {
                                    rgbIndexSelector.ReconstructLDR_BC7(altRGBIndexes[ii], reconstructedRGB);
                                    alphaIndexSelector.ReconstructLDR_BC7(altAlphaIndexes[ii], reconstructedAlpha);

                                    AggregatedError<3> altRGBAggError;
                                    AggregatedError<1> altAlphaAggError;

                                    BCCommon::ComputeErrorLDR<3>(flags, reconstructedRGB, rotatedRGB[px], altRGBAggError);
                                    BCCommon::ComputeErrorLDR<1>(flags, reconstructedAlpha, pixels[px] + alphaChannel, altAlphaAggError);

                                    MFloat altRGBError = altRGBAggError.Finalize(flags, rotatedRGBWeightsSq);
                                    MFloat altAlphaError = altAlphaAggError.Finalize(flags, rotatedAlphaWeightSq);

                                    ParallelMath::Int16CompFlag rgbBetter = ParallelMath::FloatFlagToInt16(ParallelMath::Less(altRGBError, rgbError));
                                    ParallelMath::Int16CompFlag alphaBetter = ParallelMath::FloatFlagToInt16(ParallelMath::Less(altAlphaError, alphaError));

                                    rgbError = ParallelMath::Min(altRGBError, rgbError);
                                    alphaError = ParallelMath::Min(altAlphaError, alphaError);

                                    ParallelMath::ConditionalSet(rgbIndex, rgbBetter, altRGBIndexes[ii]);
                                    ParallelMath::ConditionalSet(alphaIndex, alphaBetter, altAlphaIndexes[ii]);
                                }

                                errorRGB = errorRGB + rgbError;
                                errorA = errorA + alphaError;
                            }

                            if (refine != numRefineRounds - 1)
                            {
                                rgbRefiner.ContributeUnweightedPW(preWeightedRotatedRGB[px], rgbIndex);
                                alphaRefiner.ContributeUnweightedPW(floatPixels[px] + alphaChannel, alphaIndex);
                            }

                            if (flags & Flags::BC7_FastIndexing)
                            {
                                errorRGB = rgbAggError.Finalize(flags, rotatedRGBWeightsSq);
                                errorA = alphaAggError.Finalize(flags, rotatedAlphaWeightSq);
                            }

                            rgbIndexes[px] = rgbIndex;
                            alphaIndexes[px] = alphaIndex;
                        }

                        ParallelMath::FloatCompFlag rgbBetter = ParallelMath::Less(errorRGB, bestRGBError);
                        ParallelMath::FloatCompFlag alphaBetter = ParallelMath::Less(errorA, bestAlphaError);

                        ParallelMath::Int16CompFlag rgbBetterInt16 = ParallelMath::FloatFlagToInt16(rgbBetter);
                        ParallelMath::Int16CompFlag alphaBetterInt16 = ParallelMath::FloatFlagToInt16(alphaBetter);

                        if (ParallelMath::AnySet(rgbBetterInt16))
                        {
                            bestRGBError = ParallelMath::Min(errorRGB, bestRGBError);

                            for (int px = 0; px < 16; px++)
                                ParallelMath::ConditionalSet(bestRGBIndexes[px], rgbBetterInt16, rgbIndexes[px]);

                            for (int ep = 0; ep < 2; ep++)
                            {
                                for (int ch = 0; ch < 3; ch++)
                                    ParallelMath::ConditionalSet(bestEP[ep][ch], rgbBetterInt16, rgbEP[ep][ch]);
                            }
                        }

                        if (ParallelMath::AnySet(alphaBetterInt16))
                        {
                            bestAlphaError = ParallelMath::Min(errorA, bestAlphaError);

                            for (int px = 0; px < 16; px++)
                                ParallelMath::ConditionalSet(bestAlphaIndexes[px], alphaBetterInt16, alphaIndexes[px]);

                            for (int ep = 0; ep < 2; ep++)
                                ParallelMath::ConditionalSet(bestEP[ep][3], alphaBetterInt16, alphaEP[ep]);
                        }

                        if (refine != numRefineRounds - 1)
                        {
                            rgbRefiner.GetRefinedEndpointsLDR(rgbEP, rtn);

                            MUInt15 alphaEPTemp[2][1];
                            alphaRefiner.GetRefinedEndpointsLDR(alphaEPTemp, rtn);

                            for (int i = 0; i < 2; i++)
                                alphaEP[i] = alphaEPTemp[i][0];
                        }
                    }	// refine
                } // tweak

                MFloat combinedError = bestRGBError + bestAlphaError;

                ParallelMath::FloatCompFlag errorBetter = ParallelMath::Less(combinedError, work.m_error);
                ParallelMath::Int16CompFlag errorBetter16 = ParallelMath::FloatFlagToInt16(errorBetter);

                work.m_error = ParallelMath::Min(combinedError, work.m_error);

                ParallelMath::ConditionalSet(work.m_mode, errorBetter16, ParallelMath::MakeUInt15(mode));
                ParallelMath::ConditionalSet(work.m_u.m_isr.m_rotation, errorBetter16, ParallelMath::MakeUInt15(rotation));
                ParallelMath::ConditionalSet(work.m_u.m_isr.m_indexSelector, errorBetter16, ParallelMath::MakeUInt15(indexSelector));

                for (int px = 0; px < 16; px++)
                {
                    ParallelMath::ConditionalSet(work.m_indexes[px], errorBetter16, indexSelector ? bestAlphaIndexes[px] : bestRGBIndexes[px]);
                    ParallelMath::ConditionalSet(work.m_indexes2[px], errorBetter16, indexSelector ? bestRGBIndexes[px] : bestAlphaIndexes[px]);
                }

                for (int ep = 0; ep < 2; ep++)
                    for (int ch = 0; ch < 4; ch++)
                        ParallelMath::ConditionalSet(work.m_ep[0][ep][ch], errorBetter16, bestEP[ep][ch]);
            }
        }
    }
}

template<class T>
void cvtt::Internal::BC7Computer::Swap(T& a, T& b)
{
    T temp = a;
    a = b;
    b = temp;
}

void cvtt::Internal::BC7Computer::Pack(uint32_t flags, const PixelBlockU8* inputs, uint8_t* packedBlocks, const float channelWeights[4], const BC7EncodingPlan &encodingPlan, int numRefineRounds)
{
    MUInt15 pixels[16][4];
    MFloat floatPixels[16][4];

    for (int px = 0; px < 16; px++)
    {
        for (int ch = 0; ch < 4; ch++)
            ParallelMath::ConvertLDRInputs(inputs, px, ch, pixels[px][ch]);
    }

    for (int px = 0; px < 16; px++)
    {
        for (int ch = 0; ch < 4; ch++)
            floatPixels[px][ch] = ParallelMath::ToFloat(pixels[px][ch]);
    }

    BC67::WorkInfo work;
    memset(&work, 0, sizeof(work));

    work.m_error = ParallelMath::MakeFloat(FLT_MAX);

    {
        ParallelMath::RoundTowardNearestForScope rtn;
        TrySinglePlane(flags, pixels, floatPixels, channelWeights, encodingPlan, numRefineRounds, work, &rtn);
        TryDualPlane(flags, pixels, floatPixels, channelWeights, encodingPlan, numRefineRounds, work, &rtn);
    }

    for (int block = 0; block < ParallelMath::ParallelSize; block++)
    {
        PackingVector pv;
        pv.Init();

        ParallelMath::ScalarUInt16 mode = ParallelMath::Extract(work.m_mode, block);
        ParallelMath::ScalarUInt16 partition = ParallelMath::Extract(work.m_u.m_partition, block);
        ParallelMath::ScalarUInt16 indexSelector = ParallelMath::Extract(work.m_u.m_isr.m_indexSelector, block);

        const BC7Data::BC7ModeInfo& modeInfo = BC7Data::g_modes[mode];

        ParallelMath::ScalarUInt16 indexes[16];
        ParallelMath::ScalarUInt16 indexes2[16];
        ParallelMath::ScalarUInt16 endPoints[3][2][4];

        for (int i = 0; i < 16; i++)
        {
            indexes[i] = ParallelMath::Extract(work.m_indexes[i], block);
            if (modeInfo.m_alphaMode == BC7Data::AlphaMode_Separate)
                indexes2[i] = ParallelMath::Extract(work.m_indexes2[i], block);
        }

        for (int subset = 0; subset < 3; subset++)
        {
            for (int ep = 0; ep < 2; ep++)
            {
                for (int ch = 0; ch < 4; ch++)
                    endPoints[subset][ep][ch] = ParallelMath::Extract(work.m_ep[subset][ep][ch], block);
            }
        }

        int fixups[3] = { 0, 0, 0 };

        if (modeInfo.m_alphaMode == BC7Data::AlphaMode_Separate)
        {
            bool flipRGB = ((indexes[0] & (1 << (modeInfo.m_indexBits - 1))) != 0);
            bool flipAlpha = ((indexes2[0] & (1 << (modeInfo.m_alphaIndexBits - 1))) != 0);

            if (flipRGB)
            {
                uint16_t highIndex = (1 << modeInfo.m_indexBits) - 1;
                for (int px = 0; px < 16; px++)
                    indexes[px] = highIndex - indexes[px];
            }

            if (flipAlpha)
            {
                uint16_t highIndex = (1 << modeInfo.m_alphaIndexBits) - 1;
                for (int px = 0; px < 16; px++)
                    indexes2[px] = highIndex - indexes2[px];
            }

            if (indexSelector)
                Swap(flipRGB, flipAlpha);

            if (flipRGB)
            {
                for (int ch = 0; ch < 3; ch++)
                    Swap(endPoints[0][0][ch], endPoints[0][1][ch]);
            }
            if (flipAlpha)
                Swap(endPoints[0][0][3], endPoints[0][1][3]);

        }
        else
        {
            if (modeInfo.m_numSubsets == 2)
                fixups[1] = BC7Data::g_fixupIndexes2[partition];
            else if (modeInfo.m_numSubsets == 3)
            {
                fixups[1] = BC7Data::g_fixupIndexes3[partition][0];
                fixups[2] = BC7Data::g_fixupIndexes3[partition][1];
            }

            bool flip[3] = { false, false, false };
            for (int subset = 0; subset < modeInfo.m_numSubsets; subset++)
                flip[subset] = ((indexes[fixups[subset]] & (1 << (modeInfo.m_indexBits - 1))) != 0);

            if (flip[0] || flip[1] || flip[2])
            {
                uint16_t highIndex = (1 << modeInfo.m_indexBits) - 1;
                for (int px = 0; px < 16; px++)
                {
                    int subset = 0;
                    if (modeInfo.m_numSubsets == 2)
                        subset = (BC7Data::g_partitionMap[partition] >> px) & 1;
                    else if (modeInfo.m_numSubsets == 3)
                        subset = (BC7Data::g_partitionMap2[partition] >> (px * 2)) & 3;

                    if (flip[subset])
                        indexes[px] = highIndex - indexes[px];
                }

                int maxCH = (modeInfo.m_alphaMode == BC7Data::AlphaMode_Combined) ? 4 : 3;
                for (int subset = 0; subset < modeInfo.m_numSubsets; subset++)
                {
                    if (flip[subset])
                        for (int ch = 0; ch < maxCH; ch++)
                            Swap(endPoints[subset][0][ch], endPoints[subset][1][ch]);
                }
            }
        }

        pv.Pack(static_cast<uint8_t>(1 << mode), mode + 1);

        if (modeInfo.m_partitionBits)
            pv.Pack(partition, modeInfo.m_partitionBits);

        if (modeInfo.m_alphaMode == BC7Data::AlphaMode_Separate)
        {
            ParallelMath::ScalarUInt16 rotation = ParallelMath::Extract(work.m_u.m_isr.m_rotation, block);
            pv.Pack(rotation, 2);
        }

        if (modeInfo.m_hasIndexSelector)
            pv.Pack(indexSelector, 1);

        // Encode RGB
        for (int ch = 0; ch < 3; ch++)
        {
            for (int subset = 0; subset < modeInfo.m_numSubsets; subset++)
            {
                for (int ep = 0; ep < 2; ep++)
                {
                    ParallelMath::ScalarUInt16 epPart = endPoints[subset][ep][ch];
                    epPart >>= (8 - modeInfo.m_rgbBits);

                    pv.Pack(epPart, modeInfo.m_rgbBits);
                }
            }
        }

        // Encode alpha
        if (modeInfo.m_alphaMode != BC7Data::AlphaMode_None)
        {
            for (int subset = 0; subset < modeInfo.m_numSubsets; subset++)
            {
                for (int ep = 0; ep < 2; ep++)
                {
                    ParallelMath::ScalarUInt16 epPart = endPoints[subset][ep][3];
                    epPart >>= (8 - modeInfo.m_alphaBits);

                    pv.Pack(epPart, modeInfo.m_alphaBits);
                }
            }
        }

        // Encode parity bits
        if (modeInfo.m_pBitMode == BC7Data::PBitMode_PerSubset)
        {
            for (int subset = 0; subset < modeInfo.m_numSubsets; subset++)
            {
                ParallelMath::ScalarUInt16 epPart = endPoints[subset][0][0];
                epPart >>= (7 - modeInfo.m_rgbBits);
                epPart &= 1;

                pv.Pack(epPart, 1);
            }
        }
        else if (modeInfo.m_pBitMode == BC7Data::PBitMode_PerEndpoint)
        {
            for (int subset = 0; subset < modeInfo.m_numSubsets; subset++)
            {
                for (int ep = 0; ep < 2; ep++)
                {
                    ParallelMath::ScalarUInt16 epPart = endPoints[subset][ep][0];
                    epPart >>= (7 - modeInfo.m_rgbBits);
                    epPart &= 1;

                    pv.Pack(epPart, 1);
                }
            }
        }

        // Encode indexes
        for (int px = 0; px < 16; px++)
        {
            int bits = modeInfo.m_indexBits;
            if ((px == 0) || (px == fixups[1]) || (px == fixups[2]))
                bits--;

            pv.Pack(indexes[px], bits);
        }

        // Encode secondary indexes
        if (modeInfo.m_alphaMode == BC7Data::AlphaMode_Separate)
        {
            for (int px = 0; px < 16; px++)
            {
                int bits = modeInfo.m_alphaIndexBits;
                if (px == 0)
                    bits--;

                pv.Pack(indexes2[px], bits);
            }
        }

        pv.Flush(packedBlocks);

        packedBlocks += 16;
    }
}

void cvtt::Internal::BC7Computer::UnpackOne(PixelBlockU8 &output, const uint8_t* packedBlock)
{
    UnpackingVector pv;
    pv.Init(packedBlock);

    int mode = 8;
    for (int i = 0; i < 8; i++)
    {
        if (pv.Unpack(1) == 1)
        {
            mode = i;
            break;
        }
    }

    if (mode > 7)
    {
        for (int px = 0; px < 16; px++)
            for (int ch = 0; ch < 4; ch++)
                output.m_pixels[px][ch] = 0;

        return;
    }

    const BC7Data::BC7ModeInfo &modeInfo = BC7Data::g_modes[mode];

    int partition = 0;
    if (modeInfo.m_partitionBits)
        partition = pv.Unpack(modeInfo.m_partitionBits);

    int rotation = 0;
    if (modeInfo.m_alphaMode == BC7Data::AlphaMode_Separate)
        rotation = pv.Unpack(2);

    int indexSelector = 0;
    if (modeInfo.m_hasIndexSelector)
        indexSelector = pv.Unpack(1);

    // Resolve fixups
    int fixups[3] = { 0, 0, 0 };

    if (modeInfo.m_alphaMode != BC7Data::AlphaMode_Separate)
    {
        if (modeInfo.m_numSubsets == 2)
            fixups[1] = BC7Data::g_fixupIndexes2[partition];
        else if (modeInfo.m_numSubsets == 3)
        {
            fixups[1] = BC7Data::g_fixupIndexes3[partition][0];
            fixups[2] = BC7Data::g_fixupIndexes3[partition][1];
        }
    }

    int endPoints[3][2][4];

    // Decode RGB
    for (int ch = 0; ch < 3; ch++)
    {
        for (int subset = 0; subset < modeInfo.m_numSubsets; subset++)
        {
            for (int ep = 0; ep < 2; ep++)
                endPoints[subset][ep][ch] = (pv.Unpack(modeInfo.m_rgbBits) << (8 - modeInfo.m_rgbBits));
        }
    }

    // Decode alpha
    if (modeInfo.m_alphaMode != BC7Data::AlphaMode_None)
    {
        for (int subset = 0; subset < modeInfo.m_numSubsets; subset++)
        {
            for (int ep = 0; ep < 2; ep++)
                endPoints[subset][ep][3] = (pv.Unpack(modeInfo.m_alphaBits) << (8 - modeInfo.m_alphaBits));
        }
    }
    else
    {
        for (int subset = 0; subset < modeInfo.m_numSubsets; subset++)
        {
            for (int ep = 0; ep < 2; ep++)
                endPoints[subset][ep][3] = 255;
        }
    }

    int parityBits = 0;

    // Decode parity bits
    if (modeInfo.m_pBitMode == BC7Data::PBitMode_PerSubset)
    {
        for (int subset = 0; subset < modeInfo.m_numSubsets; subset++)
        {
            int p = pv.Unpack(1);

            for (int ep = 0; ep < 2; ep++)
            {
                for (int ch = 0; ch < 3; ch++)
                    endPoints[subset][ep][ch] |= p << (7 - modeInfo.m_rgbBits);

                if (modeInfo.m_alphaMode != BC7Data::AlphaMode_None)
                    endPoints[subset][ep][3] |= p << (7 - modeInfo.m_alphaBits);
            }
        }

        parityBits = 1;
    }
    else if (modeInfo.m_pBitMode == BC7Data::PBitMode_PerEndpoint)
    {
        for (int subset = 0; subset < modeInfo.m_numSubsets; subset++)
        {
            for (int ep = 0; ep < 2; ep++)
            {
                int p = pv.Unpack(1);

                for (int ch = 0; ch < 3; ch++)
                    endPoints[subset][ep][ch] |= p << (7 - modeInfo.m_rgbBits);

                if (modeInfo.m_alphaMode != BC7Data::AlphaMode_None)
                    endPoints[subset][ep][3] |= p << (7 - modeInfo.m_alphaBits);
            }
        }

        parityBits = 1;
    }

    // Fill endpoint bits
    for (int subset = 0; subset < modeInfo.m_numSubsets; subset++)
    {
        for (int ep = 0; ep < 2; ep++)
        {
            for (int ch = 0; ch < 3; ch++)
                endPoints[subset][ep][ch] |= (endPoints[subset][ep][ch] >> (modeInfo.m_rgbBits + parityBits));

            if (modeInfo.m_alphaMode != BC7Data::AlphaMode_None)
                endPoints[subset][ep][3] |= (endPoints[subset][ep][3] >> (modeInfo.m_alphaBits + parityBits));
        }
    }

    int indexes[16];
    int indexes2[16];

    // Decode indexes
    for (int px = 0; px < 16; px++)
    {
        int bits = modeInfo.m_indexBits;
        if ((px == 0) || (px == fixups[1]) || (px == fixups[2]))
            bits--;

        indexes[px] = pv.Unpack(bits);
    }

    // Decode secondary indexes
    if (modeInfo.m_alphaMode == BC7Data::AlphaMode_Separate)
    {
        for (int px = 0; px < 16; px++)
        {
            int bits = modeInfo.m_alphaIndexBits;
            if (px == 0)
                bits--;

            indexes2[px] = pv.Unpack(bits);
        }
    }
    else
    {
        for (int px = 0; px < 16; px++)
            indexes2[px] = 0;
    }

    const int *alphaWeights = BC7Data::g_weightTables[modeInfo.m_alphaIndexBits];
    const int *rgbWeights = BC7Data::g_weightTables[modeInfo.m_indexBits];

    // Decode each pixel
    for (int px = 0; px < 16; px++)
    {
        int rgbWeight = 0;
        int alphaWeight = 0;

        int rgbIndex = indexes[px];

        rgbWeight = rgbWeights[indexes[px]];

        if (modeInfo.m_alphaMode == BC7Data::AlphaMode_Combined)
            alphaWeight = rgbWeight;
        else if (modeInfo.m_alphaMode == BC7Data::AlphaMode_Separate)
            alphaWeight = alphaWeights[indexes2[px]];

        if (indexSelector == 1)
        {
            int temp = rgbWeight;
            rgbWeight = alphaWeight;
            alphaWeight = temp;
        }

        int pixel[4] = { 0, 0, 0, 255 };

        int subset = 0;

        if (modeInfo.m_numSubsets == 2)
            subset = (BC7Data::g_partitionMap[partition] >> px) & 1;
        else if (modeInfo.m_numSubsets == 3)
            subset = (BC7Data::g_partitionMap2[partition] >> (px * 2)) & 3;

        for (int ch = 0; ch < 3; ch++)
            pixel[ch] = ((64 - rgbWeight) * endPoints[subset][0][ch] + rgbWeight * endPoints[subset][1][ch] + 32) >> 6;

        if (modeInfo.m_alphaMode != BC7Data::AlphaMode_None)
            pixel[3] = ((64 - alphaWeight) * endPoints[subset][0][3] + alphaWeight * endPoints[subset][1][3] + 32) >> 6;

        if (rotation != 0)
        {
            int ch = rotation - 1;
            int temp = pixel[ch];
            pixel[ch] = pixel[3];
            pixel[3] = temp;
        }

        for (int ch = 0; ch < 4; ch++)
            output.m_pixels[px][ch] = static_cast<uint8_t>(pixel[ch]);
    }
}

cvtt::ParallelMath::SInt16 cvtt::Internal::BC6HComputer::QuantizeSingleEndpointElementSigned(const MSInt16 &elem2CL, int precision, const ParallelMath::RoundUpForScope* ru)
{
    assert(ParallelMath::AllSet(ParallelMath::Less(elem2CL, ParallelMath::MakeSInt16(31744))));
    assert(ParallelMath::AllSet(ParallelMath::Less(ParallelMath::MakeSInt16(-31744), elem2CL)));

    // Expand to full range
    ParallelMath::Int16CompFlag isNegative = ParallelMath::Less(elem2CL, ParallelMath::MakeSInt16(0));
    MUInt15 absElem = ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::Select(isNegative, ParallelMath::MakeSInt16(0) - elem2CL, elem2CL));

    absElem = ParallelMath::RightShift(ParallelMath::RoundAndConvertToU15(ParallelMath::ToFloat(absElem) * 32.0f / 31.0f, ru), 16 - precision);

    MSInt16 absElemS16 = ParallelMath::LosslessCast<MSInt16>::Cast(absElem);

    return ParallelMath::Select(isNegative, ParallelMath::MakeSInt16(0) - absElemS16, absElemS16);
}

cvtt::ParallelMath::UInt15 cvtt::Internal::BC6HComputer::QuantizeSingleEndpointElementUnsigned(const MUInt15 &elem, int precision, const ParallelMath::RoundUpForScope* ru)
{
    MUInt16 expandedElem = ParallelMath::RoundAndConvertToU16(ParallelMath::Min(ParallelMath::ToFloat(elem) * 64.0f / 31.0f, ParallelMath::MakeFloat(65535.0f)), ru);
    return ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::RightShift(expandedElem, 16 - precision));
}

void cvtt::Internal::BC6HComputer::UnquantizeSingleEndpointElementSigned(const MSInt16 &comp, int precision, MSInt16 &outUnquantized, MSInt16 &outUnquantizedFinished2CL)
{
    MSInt16 zero = ParallelMath::MakeSInt16(0);

    ParallelMath::Int16CompFlag negative = ParallelMath::Less(comp, zero);
    MUInt15 absComp = ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::Select(negative, MSInt16(zero - comp), comp));

    MSInt16 unq;
    MUInt15 absUnq;

    if (precision >= 16)
    {
        unq = comp;
        absUnq = absComp;
    }
    else
    {
        MSInt16 maxCompMinusOne = ParallelMath::MakeSInt16(static_cast<int16_t>((1 << (precision - 1)) - 2));
        ParallelMath::Int16CompFlag isZero = ParallelMath::Equal(comp, zero);
        ParallelMath::Int16CompFlag isMax = ParallelMath::Less(maxCompMinusOne, comp);

        absUnq = (absComp << (16 - precision)) + ParallelMath::MakeUInt15(static_cast<uint16_t>(0x4000 >> (precision - 1)));
        ParallelMath::ConditionalSet(absUnq, isZero, ParallelMath::MakeUInt15(0));
        ParallelMath::ConditionalSet(absUnq, isMax, ParallelMath::MakeUInt15(0x7fff));

        unq = ParallelMath::ConditionalNegate(negative, ParallelMath::LosslessCast<MSInt16>::Cast(absUnq));
    }

    outUnquantized = unq;

    MUInt15 funq = ParallelMath::ToUInt15(ParallelMath::RightShift(ParallelMath::XMultiply(absUnq, ParallelMath::MakeUInt15(31)), 5));

    outUnquantizedFinished2CL = ParallelMath::ConditionalNegate(negative, ParallelMath::LosslessCast<MSInt16>::Cast(funq));
}

void cvtt::Internal::BC6HComputer::UnquantizeSingleEndpointElementUnsigned(const MUInt15 &comp, int precision, MUInt16 &outUnquantized, MUInt16 &outUnquantizedFinished)
{
    MUInt16 unq = ParallelMath::LosslessCast<MUInt16>::Cast(comp);
    if (precision < 15)
    {
        MUInt15 zero = ParallelMath::MakeUInt15(0);
        MUInt15 maxCompMinusOne = ParallelMath::MakeUInt15(static_cast<uint16_t>((1 << precision) - 2));

        ParallelMath::Int16CompFlag isZero = ParallelMath::Equal(comp, zero);
        ParallelMath::Int16CompFlag isMax = ParallelMath::Less(maxCompMinusOne, comp);

        unq = (ParallelMath::LosslessCast<MUInt16>::Cast(comp) << (16 - precision)) + ParallelMath::MakeUInt16(static_cast<uint16_t>(0x8000 >> precision));

        ParallelMath::ConditionalSet(unq, isZero, ParallelMath::MakeUInt16(0));
        ParallelMath::ConditionalSet(unq, isMax, ParallelMath::MakeUInt16(0xffff));
    }

    outUnquantized = unq;
    outUnquantizedFinished = ParallelMath::ToUInt16(ParallelMath::RightShift(ParallelMath::XMultiply(unq, ParallelMath::MakeUInt15(31)), 6));
}

void cvtt::Internal::BC6HComputer::QuantizeEndpointsSigned(const MSInt16 endPoints[2][3], const MFloat floatPixelsColorSpace[16][3], const MFloat floatPixelsLinearWeighted[16][3], MAInt16 quantizedEndPoints[2][3], MUInt15 indexes[16], IndexSelectorHDR<3> &indexSelector, int fixupIndex, int precision, int indexRange, const float *channelWeights, bool fastIndexing, const ParallelMath::RoundTowardNearestForScope *rtn)
{
    MSInt16 unquantizedEP[2][3];
    MSInt16 finishedUnquantizedEP[2][3];

    {
        ParallelMath::RoundUpForScope ru;

        for (int epi = 0; epi < 2; epi++)
        {
            for (int ch = 0; ch < 3; ch++)
            {
                MSInt16 qee = QuantizeSingleEndpointElementSigned(endPoints[epi][ch], precision, &ru);
                UnquantizeSingleEndpointElementSigned(qee, precision, unquantizedEP[epi][ch], finishedUnquantizedEP[epi][ch]);
                quantizedEndPoints[epi][ch] = ParallelMath::LosslessCast<MAInt16>::Cast(qee);
            }
        }
    }

    indexSelector.Init(channelWeights, unquantizedEP, finishedUnquantizedEP, indexRange);
    indexSelector.InitHDR(indexRange, true, fastIndexing, channelWeights);

    MUInt15 halfRangeMinusOne = ParallelMath::MakeUInt15(static_cast<uint16_t>(indexRange / 2) - 1);

    MUInt15 index = fastIndexing ? indexSelector.SelectIndexHDRFast(floatPixelsColorSpace[fixupIndex], rtn) : indexSelector.SelectIndexHDRSlow(floatPixelsLinearWeighted[fixupIndex], rtn);

    ParallelMath::Int16CompFlag invert = ParallelMath::Less(halfRangeMinusOne, index);

    if (ParallelMath::AnySet(invert))
    {
        ParallelMath::ConditionalSet(index, invert, MUInt15(ParallelMath::MakeUInt15(static_cast<uint16_t>(indexRange - 1)) - index));

        indexSelector.ConditionalInvert(invert);

        for (int ch = 0; ch < 3; ch++)
        {
            MAInt16 firstEP = quantizedEndPoints[0][ch];
            MAInt16 secondEP = quantizedEndPoints[1][ch];

            quantizedEndPoints[0][ch] = ParallelMath::Select(invert, secondEP, firstEP);
            quantizedEndPoints[1][ch] = ParallelMath::Select(invert, firstEP, secondEP);
        }
    }

    indexes[fixupIndex] = index;
}

void cvtt::Internal::BC6HComputer::QuantizeEndpointsUnsigned(const MSInt16 endPoints[2][3], const MFloat floatPixelsColorSpace[16][3], const MFloat floatPixelsLinearWeighted[16][3], MAInt16 quantizedEndPoints[2][3], MUInt15 indexes[16], IndexSelectorHDR<3> &indexSelector, int fixupIndex, int precision, int indexRange, const float *channelWeights, bool fastIndexing, const ParallelMath::RoundTowardNearestForScope *rtn)
{
    MUInt16 unquantizedEP[2][3];
    MUInt16 finishedUnquantizedEP[2][3];

    {
        ParallelMath::RoundUpForScope ru;

        for (int epi = 0; epi < 2; epi++)
        {
            for (int ch = 0; ch < 3; ch++)
            {
                MUInt15 qee = QuantizeSingleEndpointElementUnsigned(ParallelMath::LosslessCast<MUInt15>::Cast(endPoints[epi][ch]), precision, &ru);
                UnquantizeSingleEndpointElementUnsigned(qee, precision, unquantizedEP[epi][ch], finishedUnquantizedEP[epi][ch]);
                quantizedEndPoints[epi][ch] = ParallelMath::LosslessCast<MAInt16>::Cast(qee);
            }
        }
    }

    indexSelector.Init(channelWeights, unquantizedEP, finishedUnquantizedEP, indexRange);
    indexSelector.InitHDR(indexRange, false, fastIndexing, channelWeights);

    MUInt15 halfRangeMinusOne = ParallelMath::MakeUInt15(static_cast<uint16_t>(indexRange / 2) - 1);

    MUInt15 index = fastIndexing ? indexSelector.SelectIndexHDRFast(floatPixelsColorSpace[fixupIndex], rtn) : indexSelector.SelectIndexHDRSlow(floatPixelsLinearWeighted[fixupIndex], rtn);

    ParallelMath::Int16CompFlag invert = ParallelMath::Less(halfRangeMinusOne, index);

    if (ParallelMath::AnySet(invert))
    {
        ParallelMath::ConditionalSet(index, invert, MUInt15(ParallelMath::MakeUInt15(static_cast<uint16_t>(indexRange - 1)) - index));

        indexSelector.ConditionalInvert(invert);

        for (int ch = 0; ch < 3; ch++)
        {
            MAInt16 firstEP = quantizedEndPoints[0][ch];
            MAInt16 secondEP = quantizedEndPoints[1][ch];

            quantizedEndPoints[0][ch] = ParallelMath::Select(invert, secondEP, firstEP);
            quantizedEndPoints[1][ch] = ParallelMath::Select(invert, firstEP, secondEP);
        }
    }

    indexes[fixupIndex] = index;
}

void cvtt::Internal::BC6HComputer::EvaluatePartitionedLegality(const MAInt16 ep0[2][3], const MAInt16 ep1[2][3], int aPrec, const int bPrec[3], bool isTransformed, MAInt16 outEncodedEPs[2][2][3], ParallelMath::Int16CompFlag& outIsLegal)
{
    ParallelMath::Int16CompFlag allLegal = ParallelMath::MakeBoolInt16(true);

    MAInt16 aSignificantMask = ParallelMath::MakeAInt16(static_cast<int16_t>((1 << aPrec) - 1));

    for (int ch = 0; ch < 3; ch++)
    {
        outEncodedEPs[0][0][ch] = ep0[0][ch];
        outEncodedEPs[0][1][ch] = ep0[1][ch];
        outEncodedEPs[1][0][ch] = ep1[0][ch];
        outEncodedEPs[1][1][ch] = ep1[1][ch];

        if (isTransformed)
        {
            for (int subset = 0; subset < 2; subset++)
            {
                for (int epi = 0; epi < 2; epi++)
                {
                    if (epi == 0 && subset == 0)
                        continue;

                    MAInt16 bReduced = (outEncodedEPs[subset][epi][ch] & aSignificantMask);

                    MSInt16 delta = ParallelMath::TruncateToPrecisionSigned(ParallelMath::LosslessCast<MSInt16>::Cast(ParallelMath::AbstractSubtract(outEncodedEPs[subset][epi][ch], outEncodedEPs[0][0][ch])), bPrec[ch]);

                    outEncodedEPs[subset][epi][ch] = ParallelMath::LosslessCast<MAInt16>::Cast(delta);

                    MAInt16 reconstructed = (ParallelMath::AbstractAdd(outEncodedEPs[subset][epi][ch], outEncodedEPs[0][0][ch]) & aSignificantMask);
                    allLegal = allLegal & ParallelMath::Equal(reconstructed, bReduced);
                }
            }
        }

        if (!ParallelMath::AnySet(allLegal))
            break;
    }

    outIsLegal = allLegal;
}

void cvtt::Internal::BC6HComputer::EvaluateSingleLegality(const MAInt16 ep[2][3], int aPrec, const int bPrec[3], bool isTransformed, MAInt16 outEncodedEPs[2][3], ParallelMath::Int16CompFlag& outIsLegal)
{
    ParallelMath::Int16CompFlag allLegal = ParallelMath::MakeBoolInt16(true);

    MAInt16 aSignificantMask = ParallelMath::MakeAInt16(static_cast<int16_t>((1 << aPrec) - 1));

    for (int ch = 0; ch < 3; ch++)
    {
        outEncodedEPs[0][ch] = ep[0][ch];
        outEncodedEPs[1][ch] = ep[1][ch];

        if (isTransformed)
        {
            MAInt16 bReduced = (outEncodedEPs[1][ch] & aSignificantMask);

            MSInt16 delta = ParallelMath::TruncateToPrecisionSigned(ParallelMath::LosslessCast<MSInt16>::Cast(ParallelMath::AbstractSubtract(outEncodedEPs[1][ch], outEncodedEPs[0][ch])), bPrec[ch]);

            outEncodedEPs[1][ch] = ParallelMath::LosslessCast<MAInt16>::Cast(delta);

            MAInt16 reconstructed = (ParallelMath::AbstractAdd(outEncodedEPs[1][ch], outEncodedEPs[0][ch]) & aSignificantMask);
            allLegal = allLegal & ParallelMath::Equal(reconstructed, bReduced);
        }
    }

    outIsLegal = allLegal;
}

void cvtt::Internal::BC6HComputer::Pack(uint32_t flags, const PixelBlockF16* inputs, uint8_t* packedBlocks, const float channelWeights[4], bool isSigned, int numTweakRounds, int numRefineRounds)
{
    if (numTweakRounds < 1)
        numTweakRounds = 1;
    else if (numTweakRounds > MaxTweakRounds)
        numTweakRounds = MaxTweakRounds;

    if (numRefineRounds < 1)
        numRefineRounds = 1;
    else if (numRefineRounds > MaxRefineRounds)
        numRefineRounds = MaxRefineRounds;

    bool fastIndexing = ((flags & cvtt::Flags::BC6H_FastIndexing) != 0);
    float channelWeightsSq[3];

    ParallelMath::RoundTowardNearestForScope rtn;

    MSInt16 pixels[16][3];
    MFloat floatPixels2CL[16][3];
    MFloat floatPixelsLinearWeighted[16][3];

    MSInt16 low15Bits = ParallelMath::MakeSInt16(32767);

    for (int ch = 0; ch < 3; ch++)
        channelWeightsSq[ch] = channelWeights[ch] * channelWeights[ch];

    for (int px = 0; px < 16; px++)
    {
        for (int ch = 0; ch < 3; ch++)
        {
            MSInt16 pixelValue;
            ParallelMath::ConvertHDRInputs(inputs, px, ch, pixelValue);

            // Convert from sign+magnitude to 2CL
            if (isSigned)
            {
                ParallelMath::Int16CompFlag negative = ParallelMath::Less(pixelValue, ParallelMath::MakeSInt16(0));
                MSInt16 magnitude = (pixelValue & low15Bits);
                ParallelMath::ConditionalSet(pixelValue, negative, ParallelMath::MakeSInt16(0) - magnitude);
                pixelValue = ParallelMath::Max(pixelValue, ParallelMath::MakeSInt16(-31743));
            }
            else
                pixelValue = ParallelMath::Max(pixelValue, ParallelMath::MakeSInt16(0));

            pixelValue = ParallelMath::Min(pixelValue, ParallelMath::MakeSInt16(31743));

            pixels[px][ch] = pixelValue;
            floatPixels2CL[px][ch] = ParallelMath::ToFloat(pixelValue);
            floatPixelsLinearWeighted[px][ch] = ParallelMath::TwosCLHalfToFloat(pixelValue) * channelWeights[ch];
        }
    }

    MFloat preWeightedPixels[16][3];

    BCCommon::PreWeightPixelsHDR<3>(preWeightedPixels, pixels, channelWeights);

    MAInt16 bestEndPoints[2][2][3];
    MUInt15 bestIndexes[16];
    MFloat bestError = ParallelMath::MakeFloat(FLT_MAX);
    MUInt15 bestMode = ParallelMath::MakeUInt15(0);
    MUInt15 bestPartition = ParallelMath::MakeUInt15(0);

    for (int px = 0; px < 16; px++)
        bestIndexes[px] = ParallelMath::MakeUInt15(0);

    for (int subset = 0; subset < 2; subset++)
        for (int epi = 0; epi < 2; epi++)
            for (int ch = 0; ch < 3; ch++)
                bestEndPoints[subset][epi][ch] = ParallelMath::MakeAInt16(0);

    UnfinishedEndpoints<3> partitionedUFEP[32][2];
    UnfinishedEndpoints<3> singleUFEP;

    // Generate UFEP for partitions
    for (int p = 0; p < 32; p++)
    {
        int partitionMask = BC7Data::g_partitionMap[p];

        EndpointSelector<3, 8> epSelectors[2];

        for (int pass = 0; pass < NumEndpointSelectorPasses; pass++)
        {
            for (int px = 0; px < 16; px++)
            {
                int subset = (partitionMask >> px) & 1;
                epSelectors[subset].ContributePass(preWeightedPixels[px], pass, ParallelMath::MakeFloat(1.0f));
            }

            for (int subset = 0; subset < 2; subset++)
                epSelectors[subset].FinishPass(pass);
        }

        for (int subset = 0; subset < 2; subset++)
            partitionedUFEP[p][subset] = epSelectors[subset].GetEndpoints(channelWeights);
    }

    // Generate UFEP for single
    {
        EndpointSelector<3, 8> epSelector;

        for (int pass = 0; pass < NumEndpointSelectorPasses; pass++)
        {
            for (int px = 0; px < 16; px++)
                epSelector.ContributePass(preWeightedPixels[px], pass, ParallelMath::MakeFloat(1.0f));

            epSelector.FinishPass(pass);
        }

        singleUFEP = epSelector.GetEndpoints(channelWeights);
    }

    for (int partitionedInt = 0; partitionedInt < 2; partitionedInt++)
    {
        bool partitioned = (partitionedInt == 1);

        for (int aPrec = BC7Data::g_maxHDRPrecision; aPrec >= 0; aPrec--)
        {
            if (!BC7Data::g_hdrModesExistForPrecision[partitionedInt][aPrec])
                continue;

            int numPartitions = partitioned ? 32 : 1;
            int numSubsets = partitioned ? 2 : 1;
            int indexBits = partitioned ? 3 : 4;
            int indexRange = (1 << indexBits);

            for (int p = 0; p < numPartitions; p++)
            {
                int partitionMask = partitioned ? BC7Data::g_partitionMap[p] : 0;

                const int MaxMetaRounds = MaxTweakRounds * MaxRefineRounds;

                MAInt16 metaEndPointsQuantized[MaxMetaRounds][2][2][3];
                MUInt15 metaIndexes[MaxMetaRounds][16];
                MFloat metaError[MaxMetaRounds][2];

                bool roundValid[MaxMetaRounds][2];

                for (int r = 0; r < MaxMetaRounds; r++)
                    for (int subset = 0; subset < 2; subset++)
                        roundValid[r][subset] = true;

                for (int subset = 0; subset < numSubsets; subset++)
                {
                    for (int tweak = 0; tweak < MaxTweakRounds; tweak++)
                    {
                        EndpointRefiner<3> refiners[2];

                        bool abortRemainingRefines = false;
                        for (int refinePass = 0; refinePass < MaxRefineRounds; refinePass++)
                        {
                            int metaRound = tweak * MaxRefineRounds + refinePass;

                            if (tweak >= numTweakRounds || refinePass >= numRefineRounds)
                                abortRemainingRefines = true;

                            if (abortRemainingRefines)
                            {
                                roundValid[metaRound][subset] = false;
                                continue;
                            }

                            MAInt16(&mrQuantizedEndPoints)[2][2][3] = metaEndPointsQuantized[metaRound];
                            MUInt15(&mrIndexes)[16] = metaIndexes[metaRound];

                            MSInt16 endPointsColorSpace[2][3];

                            if (refinePass == 0)
                            {
                                UnfinishedEndpoints<3> ufep = partitioned ? partitionedUFEP[p][subset] : singleUFEP;

                                if (isSigned)
                                    ufep.FinishHDRSigned(tweak, indexRange, endPointsColorSpace[0], endPointsColorSpace[1], &rtn);
                                else
                                    ufep.FinishHDRUnsigned(tweak, indexRange, endPointsColorSpace[0], endPointsColorSpace[1], &rtn);
                            }
                            else
                                refiners[subset].GetRefinedEndpointsHDR(endPointsColorSpace, isSigned, &rtn);

                            refiners[subset].Init(indexRange, channelWeights);

                            int fixupIndex = (subset == 0) ? 0 : BC7Data::g_fixupIndexes2[p];

                            IndexSelectorHDR<3> indexSelector;
                            if (isSigned)
                                QuantizeEndpointsSigned(endPointsColorSpace, floatPixels2CL, floatPixelsLinearWeighted, mrQuantizedEndPoints[subset], mrIndexes, indexSelector, fixupIndex, aPrec, indexRange, channelWeights, fastIndexing, &rtn);
                            else
                                QuantizeEndpointsUnsigned(endPointsColorSpace, floatPixels2CL, floatPixelsLinearWeighted, mrQuantizedEndPoints[subset], mrIndexes, indexSelector, fixupIndex, aPrec, indexRange, channelWeights, fastIndexing, &rtn);

                            if (metaRound > 0)
                            {
                                ParallelMath::Int16CompFlag anySame = ParallelMath::MakeBoolInt16(false);

                                for (int prevRound = 0; prevRound < metaRound; prevRound++)
                                {
                                    MAInt16(&prevRoundEPs)[2][3] = metaEndPointsQuantized[prevRound][subset];

                                    ParallelMath::Int16CompFlag same = ParallelMath::MakeBoolInt16(true);

                                    for (int epi = 0; epi < 2; epi++)
                                        for (int ch = 0; ch < 3; ch++)
                                            same = (same & ParallelMath::Equal(prevRoundEPs[epi][ch], mrQuantizedEndPoints[subset][epi][ch]));

                                    anySame = (anySame | same);
                                    if (ParallelMath::AllSet(anySame))
                                        break;
                                }

                                if (ParallelMath::AllSet(anySame))
                                {
                                    roundValid[metaRound][subset] = false;
                                    continue;
                                }
                            }

                            MFloat subsetError = ParallelMath::MakeFloatZero();

                            {
                                for (int px = 0; px < 16; px++)
                                {
                                    if (subset != ((partitionMask >> px) & 1))
                                        continue;

                                    MUInt15 index;
                                    if (px == fixupIndex)
                                        index = mrIndexes[px];
                                    else
                                    {
                                        index = fastIndexing ? indexSelector.SelectIndexHDRFast(floatPixels2CL[px], &rtn) : indexSelector.SelectIndexHDRSlow(floatPixelsLinearWeighted[px], &rtn);
                                        mrIndexes[px] = index;
                                    }

                                    MSInt16 reconstructed[3];
                                    if (isSigned)
                                        indexSelector.ReconstructHDRSigned(mrIndexes[px], reconstructed);
                                    else
                                        indexSelector.ReconstructHDRUnsigned(mrIndexes[px], reconstructed);

                                    subsetError = subsetError + (fastIndexing ? BCCommon::ComputeErrorHDRFast<3>(flags, reconstructed, pixels[px], channelWeightsSq) : BCCommon::ComputeErrorHDRSlow<3>(flags, reconstructed, pixels[px], channelWeightsSq));

                                    if (refinePass != numRefineRounds - 1)
                                        refiners[subset].ContributeUnweightedPW(preWeightedPixels[px], index);
                                }
                            }

                            metaError[metaRound][subset] = subsetError;
                        }
                    }
                }

                // Now we have a bunch of attempts, but not all of them will fit in the delta coding scheme
                int numMeta1 = partitioned ? MaxMetaRounds : 1;
                for (int meta0 = 0; meta0 < MaxMetaRounds; meta0++)
                {
                    if (!roundValid[meta0][0])
                        continue;

                    for (int meta1 = 0; meta1 < numMeta1; meta1++)
                    {
                        MFloat combinedError = metaError[meta0][0];
                        if (partitioned)
                        {
                            if (!roundValid[meta1][1])
                                continue;

                            combinedError = combinedError + metaError[meta1][1];
                        }

                        ParallelMath::FloatCompFlag errorBetter = ParallelMath::Less(combinedError, bestError);
                        if (!ParallelMath::AnySet(errorBetter))
                            continue;

                        ParallelMath::Int16CompFlag needsCommit = ParallelMath::FloatFlagToInt16(errorBetter);

                        // Figure out if this is encodable
                        for (int mode = 0; mode < BC7Data::g_numHDRModes; mode++)
                        {
                            const BC7Data::BC6HModeInfo &modeInfo = BC7Data::g_hdrModes[mode];

                            if (modeInfo.m_partitioned != partitioned || modeInfo.m_aPrec != aPrec)
                                continue;

                            MAInt16 encodedEPs[2][2][3];
                            ParallelMath::Int16CompFlag isLegal;
                            if (partitioned)
                                EvaluatePartitionedLegality(metaEndPointsQuantized[meta0][0], metaEndPointsQuantized[meta1][1], modeInfo.m_aPrec, modeInfo.m_bPrec, modeInfo.m_transformed, encodedEPs, isLegal);
                            else
                                EvaluateSingleLegality(metaEndPointsQuantized[meta0][0], modeInfo.m_aPrec, modeInfo.m_bPrec, modeInfo.m_transformed, encodedEPs[0], isLegal);

                            ParallelMath::Int16CompFlag isLegalAndBetter = (ParallelMath::FloatFlagToInt16(errorBetter) & isLegal);
                            if (!ParallelMath::AnySet(isLegalAndBetter))
                                continue;

                            ParallelMath::FloatCompFlag isLegalAndBetterFloat = ParallelMath::Int16FlagToFloat(isLegalAndBetter);

                            ParallelMath::ConditionalSet(bestError, isLegalAndBetterFloat, combinedError);
                            ParallelMath::ConditionalSet(bestMode, isLegalAndBetter, ParallelMath::MakeUInt15(static_cast<uint16_t>(mode)));
                            ParallelMath::ConditionalSet(bestPartition, isLegalAndBetter, ParallelMath::MakeUInt15(static_cast<uint16_t>(p)));

                            for (int subset = 0; subset < numSubsets; subset++)
                            {
                                for (int epi = 0; epi < 2; epi++)
                                {
                                    for (int ch = 0; ch < 3; ch++)
                                        ParallelMath::ConditionalSet(bestEndPoints[subset][epi][ch], isLegalAndBetter, encodedEPs[subset][epi][ch]);
                                }
                            }

                            for (int px = 0; px < 16; px++)
                            {
                                int subset = ((partitionMask >> px) & 1);
                                if (subset == 0)
                                    ParallelMath::ConditionalSet(bestIndexes[px], isLegalAndBetter, metaIndexes[meta0][px]);
                                else
                                    ParallelMath::ConditionalSet(bestIndexes[px], isLegalAndBetter, metaIndexes[meta1][px]);
                            }

                            needsCommit = ParallelMath::AndNot(needsCommit, isLegalAndBetter);
                            if (!ParallelMath::AnySet(needsCommit))
                                break;
                        }
                    }
                }
            }
        }
    }

    // At this point, everything should be set
    for (int block = 0; block < ParallelMath::ParallelSize; block++)
    {
        ParallelMath::ScalarUInt16 mode = ParallelMath::Extract(bestMode, block);
        ParallelMath::ScalarUInt16 partition = ParallelMath::Extract(bestPartition, block);
        int32_t eps[2][2][3];
        ParallelMath::ScalarUInt16 indexes[16];

        const BC7Data::BC6HModeInfo& modeInfo = BC7Data::g_hdrModes[mode];

        BC6H_IO::WriteFunc_t writeFunc = BC6H_IO::g_writeFuncs[mode];

        const int headerBits = modeInfo.m_partitioned ? 82 : 65;

        for (int subset = 0; subset < 2; subset++)
        {
            for (int epi = 0; epi < 2; epi++)
            {
                for (int ch = 0; ch < 3; ch++)
                    eps[subset][epi][ch] = ParallelMath::Extract(bestEndPoints[subset][epi][ch], block);
            }
        }

        for (int px = 0; px < 16; px++)
            indexes[px] = ParallelMath::Extract(bestIndexes[px], block);

        uint16_t modeID = modeInfo.m_modeID;

        PackingVector pv;

        {
            uint32_t header[3];
            writeFunc(header, modeID, partition,
                eps[0][0][0], eps[0][1][0], eps[1][0][0], eps[1][1][0],
                eps[0][0][1], eps[0][1][1], eps[1][0][1], eps[1][1][1],
                eps[0][0][2], eps[0][1][2], eps[1][0][2], eps[1][1][2]
            );

            pv.InitPacked(header, headerBits);
        }

        int fixupIndex1 = 0;
        int indexBits = 4;
        if (modeInfo.m_partitioned)
        {
            fixupIndex1 = BC7Data::g_fixupIndexes2[partition];
            indexBits = 3;
        }

        for (int px = 0; px < 16; px++)
        {
            ParallelMath::ScalarUInt16 index = ParallelMath::Extract(bestIndexes[px], block);
            if (px == 0 || px == fixupIndex1)
                pv.Pack(index, indexBits - 1);
            else
                pv.Pack(index, indexBits);
        }

        pv.Flush(packedBlocks + 16 * block);
    }
}

void cvtt::Internal::BC6HComputer::SignExtendSingle(int &v, int bits)
{
    if (v & (1 << (bits - 1)))
        v |= -(1 << bits);
}

void cvtt::Internal::BC6HComputer::UnpackOne(PixelBlockF16 &output, const uint8_t *pBC, bool isSigned)
{
    UnpackingVector pv;
    pv.Init(pBC);

    int numModeBits = 2;
    int modeBits = pv.Unpack(2);
    if (modeBits != 0 && modeBits != 1)
    {
        modeBits |= pv.Unpack(3) << 2;
        numModeBits += 3;
    }

    int mode = -1;
    for (int possibleMode = 0; possibleMode < BC7Data::g_numHDRModes; possibleMode++)
    {
        if (BC7Data::g_hdrModes[possibleMode].m_modeID == modeBits)
        {
            mode = possibleMode;
            break;
        }
    }

    if (mode < 0)
    {
        for (int px = 0; px < 16; px++)
        {
            for (int ch = 0; ch < 3; ch++)
                output.m_pixels[px][ch] = 0;
            output.m_pixels[px][3] = 0x3c00;	// 1.0
        }
        return;
    }

    const BC7Data::BC6HModeInfo& modeInfo = BC7Data::g_hdrModes[mode];
    const int headerBits = modeInfo.m_partitioned ? 82 : 65;
    const BC6H_IO::ReadFunc_t readFunc = BC6H_IO::g_readFuncs[mode];

    uint16_t partition = 0;
    int32_t eps[2][2][3];

    for (int subset = 0; subset < 2; subset++)
        for (int epi = 0; epi < 2; epi++)
            for (int ch = 0; ch < 3; ch++)
                eps[subset][epi][ch] = 0;

    {
        uint32_t header[3];
        uint16_t codedEPs[2][2][3];
        pv.UnpackStart(header, headerBits);

        readFunc(header, partition,
            codedEPs[0][0][0], codedEPs[0][1][0], codedEPs[1][0][0], codedEPs[1][1][0],
            codedEPs[0][0][1], codedEPs[0][1][1], codedEPs[1][0][1], codedEPs[1][1][1],
            codedEPs[0][0][2], codedEPs[0][1][2], codedEPs[1][0][2], codedEPs[1][1][2]
        );

        for (int subset = 0; subset < 2; subset++)
            for (int epi = 0; epi < 2; epi++)
                for (int ch = 0; ch < 3; ch++)
                    eps[subset][epi][ch] = codedEPs[subset][epi][ch];
    }

    uint16_t modeID = modeInfo.m_modeID;

    int fixupIndex1 = 0;
    int indexBits = 4;
    int numSubsets = 1;
    if (modeInfo.m_partitioned)
    {
        fixupIndex1 = BC7Data::g_fixupIndexes2[partition];
        indexBits = 3;
        numSubsets = 2;
    }

    int indexes[16];
    for (int px = 0; px < 16; px++)
    {
        if (px == 0 || px == fixupIndex1)
            indexes[px] = pv.Unpack(indexBits - 1);
        else
            indexes[px] = pv.Unpack(indexBits);
    }

    if (modeInfo.m_partitioned)
    {
        for (int ch = 0; ch < 3; ch++)
        {
            if (isSigned)
                SignExtendSingle(eps[0][0][ch], modeInfo.m_aPrec);
            if (modeInfo.m_transformed || isSigned)
            {
                SignExtendSingle(eps[0][1][ch], modeInfo.m_bPrec[ch]);
                SignExtendSingle(eps[1][0][ch], modeInfo.m_bPrec[ch]);
                SignExtendSingle(eps[1][1][ch], modeInfo.m_bPrec[ch]);
            }
        }
    }
    else
    {
        for (int ch = 0; ch < 3; ch++)
        {
            if (isSigned)
                SignExtendSingle(eps[0][0][ch], modeInfo.m_aPrec);
            if (modeInfo.m_transformed || isSigned)
                SignExtendSingle(eps[0][1][ch], modeInfo.m_bPrec[ch]);
        }
    }

    int aPrec = modeInfo.m_aPrec;

    if (modeInfo.m_transformed)
    {
        for (int ch = 0; ch < 3; ch++)
        {
            int wrapMask = (1 << aPrec) - 1;

            eps[0][1][ch] = ((eps[0][0][ch] + eps[0][1][ch]) & wrapMask);
            if (isSigned)
                SignExtendSingle(eps[0][1][ch], aPrec);

            if (modeInfo.m_partitioned)
            {
                eps[1][0][ch] = ((eps[0][0][ch] + eps[1][0][ch]) & wrapMask);
                eps[1][1][ch] = ((eps[0][0][ch] + eps[1][1][ch]) & wrapMask);

                if (isSigned)
                {
                    SignExtendSingle(eps[1][0][ch], aPrec);
                    SignExtendSingle(eps[1][1][ch], aPrec);
                }
            }
        }
    }

    // Unquantize endpoints
    for (int subset = 0; subset < numSubsets; subset++)
    {
        for (int epi = 0; epi < 2; epi++)
        {
            for (int ch = 0; ch < 3; ch++)
            {
                int &v = eps[subset][epi][ch];

                if (isSigned)
                {
                    if (aPrec >= 16)
                    {
                        // Nothing
                    }
                    else
                    {
                        bool s = false;
                        int comp = v;
                        if (v < 0)
                        {
                            s = true;
                            comp = -comp;
                        }

                        int unq = 0;
                        if (comp == 0)
                            unq = 0;
                        else if (comp >= ((1 << (aPrec - 1)) - 1))
                            unq = 0x7fff;
                        else
                            unq = ((comp << 15) + 0x4000) >> (aPrec - 1);

                        if (s)
                            unq = -unq;

                        v = unq;
                    }
                }
                else
                {
                    if (aPrec >= 15)
                    {
                        // Nothing
                    }
                    else if (v == 0)
                    {
                        // Nothing
                    }
                    else if (v == ((1 << aPrec) - 1))
                        v = 0xffff;
                    else
                        v = ((v << 16) + 0x8000) >> aPrec;
                }
            }
        }
    }

    const int *weights = BC7Data::g_weightTables[indexBits];

    for (int px = 0; px < 16; px++)
    {
        int subset = 0;
        if (modeInfo.m_partitioned)
            subset = (BC7Data::g_partitionMap[partition] >> px) & 1;

        int w = weights[indexes[px]];
        for (int ch = 0; ch < 3; ch++)
        {
            int comp = ((64 - w) * eps[subset][0][ch] + w * eps[subset][1][ch] + 32) >> 6;

            if (isSigned)
            {
                if (comp < 0)
                    comp = -(((-comp) * 31) >> 5);
                else
                    comp = (comp * 31) >> 5;

                int s = 0;
                if (comp < 0)
                {
                    s = 0x8000;
                    comp = -comp;
                }

                output.m_pixels[px][ch] = static_cast<uint16_t>(s | comp);
            }
            else
            {
                comp = (comp * 31) >> 6;
                output.m_pixels[px][ch] = static_cast<uint16_t>(comp);
            }
        }
        output.m_pixels[px][3] = 0x3c00;	// 1.0
    }
}

void cvtt::Kernels::ConfigureBC7EncodingPlanFromQuality(BC7EncodingPlan &encodingPlan, int quality)
{
    static const int kMaxQuality = 100;

    if (quality < 1)
        quality = 1;
    else if (quality > kMaxQuality)
        quality = kMaxQuality;

    const int numRGBModes = cvtt::Tables::BC7Prio::g_bc7NumPrioCodesRGB * quality / kMaxQuality;
    const int numRGBAModes = cvtt::Tables::BC7Prio::g_bc7NumPrioCodesRGBA * quality / kMaxQuality;

    const uint16_t *prioLists[] = { cvtt::Tables::BC7Prio::g_bc7PrioCodesRGB, cvtt::Tables::BC7Prio::g_bc7PrioCodesRGBA };
    const int prioListSizes[] = { numRGBModes, numRGBAModes };

    BC7FineTuningParams ftParams;
    memset(&ftParams, 0, sizeof(ftParams));

    for (int listIndex = 0; listIndex < 2; listIndex++)
    {
        int prioListSize = prioListSizes[listIndex];
        const uint16_t *prioList = prioLists[listIndex];

        for (int prioIndex = 0; prioIndex < prioListSize; prioIndex++)
        {
            const uint16_t packedMode = prioList[prioIndex];

            uint8_t seedPoints = static_cast<uint8_t>(cvtt::Tables::BC7Prio::UnpackSeedPointCount(packedMode));
            int mode = cvtt::Tables::BC7Prio::UnpackMode(packedMode);

            switch (mode)
            {
            case 0:
                ftParams.mode0SP[cvtt::Tables::BC7Prio::UnpackPartition(packedMode)] = seedPoints;
                break;
            case 1:
                ftParams.mode1SP[cvtt::Tables::BC7Prio::UnpackPartition(packedMode)] = seedPoints;
                break;
            case 2:
                ftParams.mode2SP[cvtt::Tables::BC7Prio::UnpackPartition(packedMode)] = seedPoints;
                break;
            case 3:
                ftParams.mode3SP[cvtt::Tables::BC7Prio::UnpackPartition(packedMode)] = seedPoints;
                break;
            case 4:
                ftParams.mode4SP[cvtt::Tables::BC7Prio::UnpackRotation(packedMode)][cvtt::Tables::BC7Prio::UnpackIndexSelector(packedMode)] = seedPoints;
                break;
            case 5:
                ftParams.mode5SP[cvtt::Tables::BC7Prio::UnpackRotation(packedMode)] = seedPoints;
                break;
            case 6:
                ftParams.mode6SP = seedPoints;
                break;
            case 7:
                ftParams.mode7SP[cvtt::Tables::BC7Prio::UnpackPartition(packedMode)] = seedPoints;
                break;
            }
        }
    }

    ConfigureBC7EncodingPlanFromFineTuningParams(encodingPlan, ftParams);
}

// Generates a BC7 encoding plan from fine-tuning parameters.
bool cvtt::Kernels::ConfigureBC7EncodingPlanFromFineTuningParams(BC7EncodingPlan &encodingPlan, const BC7FineTuningParams &params)
{
    memset(&encodingPlan, 0, sizeof(encodingPlan));

    // Mode 0
    for (int partition = 0; partition < 16; partition++)
    {
        uint8_t sp = params.mode0SP[partition];
        if (sp == 0)
            continue;

        encodingPlan.mode0PartitionEnabled |= static_cast<uint16_t>(1) << partition;

        for (int subset = 0; subset < 3; subset++)
        {
            int shape = cvtt::Internal::BC7Data::g_shapes3[partition][subset];
            encodingPlan.seedPointsForShapeRGB[shape] = std::max(encodingPlan.seedPointsForShapeRGB[shape], sp);
        }
    }

    // Mode 1
    for (int partition = 0; partition < 64; partition++)
    {
        uint8_t sp = params.mode1SP[partition];
        if (sp == 0)
            continue;

        encodingPlan.mode1PartitionEnabled |= static_cast<uint64_t>(1) << partition;

        for (int subset = 0; subset < 2; subset++)
        {
            int shape = cvtt::Internal::BC7Data::g_shapes2[partition][subset];
            encodingPlan.seedPointsForShapeRGB[shape] = std::max(encodingPlan.seedPointsForShapeRGB[shape], sp);
        }
    }

    // Mode 2
    for (int partition = 0; partition < 64; partition++)
    {
        uint8_t sp = params.mode2SP[partition];
        if (sp == 0)
            continue;

        encodingPlan.mode2PartitionEnabled |= static_cast<uint64_t>(1) << partition;

        for (int subset = 0; subset < 3; subset++)
        {
            int shape = cvtt::Internal::BC7Data::g_shapes3[partition][subset];
            encodingPlan.seedPointsForShapeRGB[shape] = std::max(encodingPlan.seedPointsForShapeRGB[shape], sp);
        }
    }

    // Mode 3
    for (int partition = 0; partition < 64; partition++)
    {
        uint8_t sp = params.mode3SP[partition];
        if (sp == 0)
            continue;

        encodingPlan.mode3PartitionEnabled |= static_cast<uint64_t>(1) << partition;

        for (int subset = 0; subset < 2; subset++)
        {
            int shape = cvtt::Internal::BC7Data::g_shapes2[partition][subset];
            encodingPlan.seedPointsForShapeRGB[shape] = std::max(encodingPlan.seedPointsForShapeRGB[shape], sp);
        }
    }

    // Mode 4
    for (int rotation = 0; rotation < 4; rotation++)
    {
        for (int indexMode = 0; indexMode < 2; indexMode++)
            encodingPlan.mode4SP[rotation][indexMode] = params.mode4SP[rotation][indexMode];
    }

    // Mode 5
    for (int rotation = 0; rotation < 4; rotation++)
        encodingPlan.mode5SP[rotation] = params.mode5SP[rotation];

    // Mode 6
    {
        uint8_t sp = params.mode6SP;
        if (sp != 0)
        {
            encodingPlan.mode6Enabled = true;

            int shape = cvtt::Internal::BC7Data::g_shapes1[0][0];
            encodingPlan.seedPointsForShapeRGBA[shape] = std::max(encodingPlan.seedPointsForShapeRGBA[shape], sp);
        }
    }

    // Mode 7
    for (int partition = 0; partition < 64; partition++)
    {
        uint8_t sp = params.mode7SP[partition];
        if (sp == 0)
            continue;

        encodingPlan.mode7RGBAPartitionEnabled |= static_cast<uint64_t>(1) << partition;

        for (int subset = 0; subset < 2; subset++)
        {
            int shape = cvtt::Internal::BC7Data::g_shapes2[partition][subset];
            encodingPlan.seedPointsForShapeRGBA[shape] = std::max(encodingPlan.seedPointsForShapeRGBA[shape], sp);
        }
    }

    for (int i = 0; i < BC7EncodingPlan::kNumRGBShapes; i++)
    {
        if (encodingPlan.seedPointsForShapeRGB[i] > 0)
        {
            encodingPlan.rgbShapeList[encodingPlan.rgbNumShapesToEvaluate] = i;
            encodingPlan.rgbNumShapesToEvaluate++;
        }
    }

    for (int i = 0; i < BC7EncodingPlan::kNumRGBAShapes; i++)
    {
        if (encodingPlan.seedPointsForShapeRGBA[i] > 0)
        {
            encodingPlan.rgbaShapeList[encodingPlan.rgbaNumShapesToEvaluate] = i;
            encodingPlan.rgbaNumShapesToEvaluate++;
        }
    }

    encodingPlan.mode7RGBPartitionEnabled = (encodingPlan.mode7RGBAPartitionEnabled & ~encodingPlan.mode3PartitionEnabled);

    return true;
}

#endif
