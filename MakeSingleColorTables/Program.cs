using System;
using System.Collections.Generic;
using System.IO;

namespace MakeSingleColorTables
{
    class Program
    {
        static int BitExpand(int v, int bits)
        {
            v <<= (8 - bits);
            return (v | (v >> bits));
        }

        static int BitExpandP(int v, int bits, int parityBit)
        {
            v <<= (8 - bits);
            v |= (parityBit << (7 - bits));
            v |= (v >> (bits + 1));
            return v;
        }

        static int[] aWeight2 = { 0, 21, 43, 64 };
        static int[] aWeight3 = { 0, 9, 18, 27, 37, 46, 55, 64 };
        static int[] aWeight4 = { 0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64 };

        static void EmitTableBC7(StreamWriter w, int bits, int parityBits, int parityBitMin, int parityBitMax, int targetIndex, int maxIndex, string name)
        {
            int parityBitsCombined = parityBitMin;
            if (parityBits == 2)
                parityBitsCombined += (parityBitMax << 1);

            w.WriteLine("Table " + name + "=");
            w.WriteLine("{");
            w.WriteLine("    " + targetIndex + ",");
            w.WriteLine("    " + parityBitsCombined + ",");
            w.WriteLine("    {");

            int epRange = 1 << bits;

            for (int i = 0; i < 256; i++)
            {
                if (i % 8 == 0)
                    w.Write("        ");

                double bestError = double.MaxValue;
                int bestMin = 0;
                int bestMax = 0;
                int bestActualColor = 0;

                int[] weightTable = null;
                if (maxIndex == 3)
                    weightTable = aWeight2;
                else if (maxIndex == 7)
                    weightTable = aWeight3;
                else if (maxIndex == 15)
                    weightTable = aWeight4;

                for (int min = 0; min < epRange; min++)
                {
                    int minExpanded = (parityBits != 0) ? BitExpandP(min, bits, parityBitMin) : BitExpand(min, bits);

                    for (int max = 0; max < epRange; max++)
                    {
                        int maxExpanded = (parityBits != 0) ? BitExpandP(max, bits, parityBitMax) : BitExpand(max, bits);

                        int interpolated = (((64 - weightTable[targetIndex]) * minExpanded + weightTable[targetIndex] * maxExpanded + 32) >> 6);

                        double delta = interpolated - i;

                        double error = delta * delta;

                        if (error < bestError)
                        {
                            bestError = error;
                            bestActualColor = interpolated;
                            bestMin = minExpanded;
                            bestMax = maxExpanded;
                        }
                    }
                }

                w.Write("{ " + bestMin.ToString() + ", " + bestMax.ToString() + ", " + bestActualColor.ToString() + " },");
                if (i % 8 == 7)
                    w.WriteLine();
                else
                    w.Write(" ");
            }

            w.WriteLine("    }");
            w.WriteLine("};");
            w.WriteLine();
        }

        static void EmitTable(StreamWriter w, int bits, int maxIndex, double paranoia, string name)
        {
            w.WriteLine("TableEntry " + name + "[256] =");
            w.WriteLine("{");

            int epRange = 1 << bits;

            for (int i = 0; i < 256; i++)
            {
                if (i % 8 == 0)
                    w.Write("    ");

                double bestError = double.MaxValue;
                int bestSpan = 255;
                int bestMin = 0;
                int bestMax = 0;
                int bestActualColor = 0;

                for (int min = 0; min < epRange; min++)
                {
                    int minExpanded = BitExpand(min, bits);

                    for (int max = 0; max < epRange; max++)
                    {
                        int maxExpanded = BitExpand(max, bits);

                        int interpolated = (minExpanded * (maxIndex - 1) + maxExpanded) / maxIndex;
                        int epSpan = Math.Abs(minExpanded - maxExpanded);

                        double delta = Math.Abs(interpolated - i) + epSpan * paranoia;

                        double error = delta * delta;

                        if (error < bestError || (error == bestError && epSpan < bestSpan))
                        {
                            bestError = error;
                            bestSpan = epSpan;
                            bestActualColor = interpolated;
                            bestMin = minExpanded;
                            bestMax = maxExpanded;
                        }
                    }
                }

                w.Write("{ " + bestMin.ToString() + ", " + bestMax.ToString() + ", " + bestActualColor.ToString() + ", " + bestSpan.ToString() + " },");
                if (i % 8 == 7)
                    w.WriteLine();
                else
                    w.Write(" ");
            }

            w.WriteLine("};");
            w.WriteLine();
        }

        static void MakeETC2AlphaRoundingTables(string path)
        {
            int numRounders = 13;

            int[] etc2alphatable =
            {
                2, 5, 8, 14,
                2, 6, 9, 12,
                1, 4, 7, 12,
                1, 3, 5, 12,
                2, 5, 7, 11,
                2, 6, 8, 10,
                3, 6, 7, 10,
                2, 4, 7, 10,
                1, 5, 7, 9,
                1, 4, 7, 9,
                1, 3, 7, 9,
                1, 4, 6, 9,
                2, 3, 6, 9,
                0, 1, 2, 9,
                3, 5, 7, 8,
                2, 4, 6, 8,
            };

            using (StreamWriter w = new StreamWriter(path))
            {
                w.WriteLine("#pragma once");
                w.WriteLine("#include <stdint.h>");
                w.WriteLine();
                w.WriteLine("// This file is generated by the MakeSingleColorTables app.  Do not edit this file manually.");
                w.WriteLine();

                w.WriteLine("namespace cvtt { namespace Tables { namespace ETC2 {");
                w.WriteLine("    const int g_alphaRoundingTableWidth = " + numRounders.ToString() + ";");
                w.WriteLine("    const uint8_t g_alphaRoundingTables[16][" + numRounders.ToString() + "] =");
                w.WriteLine("    {");

                for (int table = 0; table < 16; table++)
                {
                    w.Write("        { ");

                    int baseIndex = table * 4;
                    for (int rounder = 0; rounder < numRounders; rounder++)
                    {
                        int bestIndex = 0;
                        int bestDistance = 9999;

                        for (int index = 0; index < 4; index++)
                        {
                            int absDiff = Math.Abs(rounder - etc2alphatable[baseIndex + index]);
                            if (absDiff < bestDistance)
                            {
                                bestDistance = absDiff;
                                bestIndex = index;
                            }
                        }

                        if (rounder != 0)
                            w.Write(", ");

                        w.Write(bestIndex.ToString());
                    }

                    w.WriteLine(" },");
                }

                w.WriteLine("    };");
                w.WriteLine("}}}");
            }
        }

        static void ConvertToFakeBT709(out double y, out double u, out double v, double pr, double pg, double pb)
        {
            double r = pr;
            double g = pg;
            double b = pb;

            y = r * 0.368233989135369 + g * 1.23876274963149 + b * 0.125054068802017;
            u = r * 0.5f - g * 0.4541529 - b * 0.04584709;
            v = r * -0.081014709086133 - g * 0.272538676238785 + b * 0.353553390593274;
        }

        static void EmitFakeBT709RoundingTable(StreamWriter sw, int tableResolution)
        {
            for (int r = 0; r < tableResolution; r++)
            {
                for (int g = 0; g < tableResolution; g++)
                {
                    sw.Write("        ");

                    for (int b = 0; b < tableResolution; b++)
                    {
                        double y, u, v;

                        ConvertToFakeBT709(out y, out u, out v, r, g, b);

                        double bestDiff = double.MaxValue;
                        int bestOctant = 0;
                        for (int compareOctant = 0; compareOctant < 8; compareOctant++)
                        {
                            double or = ((compareOctant & 1) == 0) ? 0 : tableResolution;
                            double og = ((compareOctant & 2) == 0) ? 0 : tableResolution;
                            double ob = ((compareOctant & 4) == 0) ? 0 : tableResolution;

                            double oy, ou, ov;
                            ConvertToFakeBT709(out oy, out ou, out ov, or, og, ob);

                            double dy = oy - y;
                            double du = ou - u;
                            double dv = ov - v;

                            double error = dy * dy + du * du + dv * dv;
                            if (error < bestDiff)
                            {
                                bestDiff = error;
                                bestOctant = compareOctant;
                            }
                        }

                        sw.Write(bestOctant);
                        sw.Write(", ");
                    }
                    sw.WriteLine();
                }
                sw.WriteLine();
            }
        }

        static void MakeFakeBT709RoundingTables(string path)
        {
            using (StreamWriter w = new StreamWriter(path))
            {
                w.WriteLine("#pragma once");
                w.WriteLine("#include <stdint.h>");
                w.WriteLine();
                w.WriteLine("// This file is generated by the MakeSingleColorTables app.  Do not edit this file manually.");
                w.WriteLine();

                w.WriteLine("namespace cvtt { namespace Tables { namespace FakeBT709 {");

                for (int r = 16; r <= 16; r++)
                {
                    w.WriteLine("    const uint8_t g_rounding" + r.ToString() + "[] =");
                    w.WriteLine("    {");

                    EmitFakeBT709RoundingTable(w, r);

                    w.WriteLine("    };");
                }
                w.WriteLine("}}}");
            }
        }

        static void Main(string[] args)
        {
            string[] filenames = { "ConvectionKernels_BC7_SingleColor.h", "ConvectionKernels_S3TC_SingleColor.h" };

            for (int i = 0; i < 2; i++)
            {
                using (StreamWriter w = new StreamWriter(filenames[i]))
                {
                    bool bc7 = (i == 0);

                    w.WriteLine("#pragma once");
                    w.WriteLine("#include <stdint.h>");
                    w.WriteLine();
                    w.WriteLine("// This file is generated by the MakeSingleColorTables app.  Do not edit this file manually.");
                    w.WriteLine();

                    if (bc7)
                        w.WriteLine("namespace cvtt { namespace Tables { namespace BC7SC {");
                    else
                        w.WriteLine("namespace cvtt { namespace Tables { namespace S3TCSC {");

                    w.WriteLine();
                    w.WriteLine("struct TableEntry");
                    w.WriteLine("{");
                    w.WriteLine("    uint8_t m_min;");
                    w.WriteLine("    uint8_t m_max;");
                    w.WriteLine("    uint8_t m_actualColor;");
                    if (!bc7)
                        w.WriteLine("    uint8_t m_span;");
                    w.WriteLine("};");
                    w.WriteLine();

                    if (bc7)
                    {
                        w.WriteLine("struct Table");
                        w.WriteLine("{");
                        w.WriteLine("    uint8_t m_index;");
                        if (bc7)
                            w.WriteLine("    uint8_t m_pBits;");
                        w.WriteLine("    TableEntry m_entries[256];");
                        w.WriteLine("};");
                        w.WriteLine();

                        // Mode 0: 5-bit endpoints, 2 P-bits, 3-bit indexes
                        EmitTableBC7(w, 4, 2, 0, 0, 1, 7, "g_mode0_p00_i1");
                        EmitTableBC7(w, 4, 2, 0, 0, 2, 7, "g_mode0_p00_i2");
                        EmitTableBC7(w, 4, 2, 0, 0, 3, 7, "g_mode0_p00_i3");
                        EmitTableBC7(w, 4, 2, 0, 1, 1, 7, "g_mode0_p01_i1");
                        EmitTableBC7(w, 4, 2, 0, 1, 2, 7, "g_mode0_p01_i2");
                        EmitTableBC7(w, 4, 2, 0, 1, 3, 7, "g_mode0_p01_i3");
                        EmitTableBC7(w, 4, 2, 1, 0, 1, 7, "g_mode0_p10_i1");
                        EmitTableBC7(w, 4, 2, 1, 0, 2, 7, "g_mode0_p10_i2");
                        EmitTableBC7(w, 4, 2, 1, 1, 3, 7, "g_mode0_p10_i3");
                        EmitTableBC7(w, 4, 2, 1, 1, 1, 7, "g_mode0_p11_i1");
                        EmitTableBC7(w, 4, 2, 1, 1, 2, 7, "g_mode0_p11_i2");
                        EmitTableBC7(w, 4, 2, 1, 1, 3, 7, "g_mode0_p11_i3");

                        // Mode 1: 6-bit endpoints, 1 P-bit, 3-bit indexes
                        EmitTableBC7(w, 6, 1, 0, 0, 1, 7, "g_mode1_p0_i1");
                        EmitTableBC7(w, 6, 1, 0, 0, 2, 7, "g_mode1_p0_i2");
                        EmitTableBC7(w, 6, 1, 0, 0, 3, 7, "g_mode1_p0_i3");
                        EmitTableBC7(w, 6, 1, 1, 1, 1, 7, "g_mode1_p1_i1");
                        EmitTableBC7(w, 6, 1, 1, 1, 2, 7, "g_mode1_p1_i2");
                        EmitTableBC7(w, 6, 1, 1, 1, 3, 7, "g_mode1_p1_i3");

                        // Mode 2: 5-bit endpoints, 0 P-bits, 2-bit indexes
                        EmitTableBC7(w, 5, 0, 0, 0, 1, 3, "g_mode2");

                        // Mode 3: 7-bit endpoints, 1 P-bit, 2-bit indexes
                        EmitTableBC7(w, 7, 1, 0, 0, 1, 3, "g_mode3_p0");
                        EmitTableBC7(w, 7, 1, 1, 1, 1, 3, "g_mode3_p1");

                        // Mode 4: 5-bit RGB endpoints, 6-bit alpha endpoints, no P-bits, 2 or 3-bit indexes
                        EmitTableBC7(w, 5, 0, 0, 0, 1, 3, "g_mode4_rgb_low");
                        EmitTableBC7(w, 5, 0, 0, 0, 1, 7, "g_mode4_rgb_high_i1");
                        EmitTableBC7(w, 5, 0, 0, 0, 2, 7, "g_mode4_rgb_high_i2");
                        EmitTableBC7(w, 5, 0, 0, 0, 3, 7, "g_mode4_rgb_high_i3");
                        EmitTableBC7(w, 6, 0, 0, 0, 1, 3, "g_mode4_a_low");
                        EmitTableBC7(w, 6, 0, 0, 0, 1, 7, "g_mode4_a_high_i1");
                        EmitTableBC7(w, 6, 0, 0, 0, 2, 7, "g_mode4_a_high_i2");
                        EmitTableBC7(w, 6, 0, 0, 0, 3, 7, "g_mode4_a_high_i3");

                        // Mode 5: 7-bit RGB endpoints, 8-bit alpha endpoints (omit), no P-bits, 2-bit indexes
                        EmitTableBC7(w, 7, 0, 0, 0, 1, 3, "g_mode5_rgb_low");

                        // Mode 6: 7-bit RGB endpoints, 1 P-bit, 4-bit indexes
                        EmitTableBC7(w, 7, 1, 0, 0, 1, 15, "g_mode6_p0_i1");
                        EmitTableBC7(w, 7, 1, 0, 0, 2, 15, "g_mode6_p0_i2");
                        EmitTableBC7(w, 7, 1, 0, 0, 3, 15, "g_mode6_p0_i3");
                        EmitTableBC7(w, 7, 1, 0, 0, 4, 15, "g_mode6_p0_i4");
                        EmitTableBC7(w, 7, 1, 0, 0, 5, 15, "g_mode6_p0_i5");
                        EmitTableBC7(w, 7, 1, 0, 0, 6, 15, "g_mode6_p0_i6");
                        EmitTableBC7(w, 7, 1, 0, 0, 7, 15, "g_mode6_p0_i7");
                        EmitTableBC7(w, 7, 1, 1, 1, 1, 15, "g_mode6_p1_i1");
                        EmitTableBC7(w, 7, 1, 1, 1, 2, 15, "g_mode6_p1_i2");
                        EmitTableBC7(w, 7, 1, 1, 1, 3, 15, "g_mode6_p1_i3");
                        EmitTableBC7(w, 7, 1, 1, 1, 4, 15, "g_mode6_p1_i4");
                        EmitTableBC7(w, 7, 1, 1, 1, 5, 15, "g_mode6_p1_i5");
                        EmitTableBC7(w, 7, 1, 1, 1, 6, 15, "g_mode6_p1_i6");
                        EmitTableBC7(w, 7, 1, 1, 1, 7, 15, "g_mode6_p1_i7");

                        // Mode 7: 5-bit RGB endpoints, 2 P-bits, 2-bit indexes
                        EmitTableBC7(w, 7, 2, 0, 0, 1, 3, "g_mode7_p00");
                        EmitTableBC7(w, 7, 2, 0, 1, 1, 3, "g_mode7_p01");
                        EmitTableBC7(w, 7, 2, 1, 0, 1, 3, "g_mode7_p10");
                        EmitTableBC7(w, 7, 2, 1, 1, 1, 3, "g_mode7_p11");
                    }
                    else
                    {
                        EmitTable(w, 5, 3, 0.0, "g_singleColor5_3");
                        EmitTable(w, 6, 3, 0.0, "g_singleColor6_3");
                        EmitTable(w, 5, 2, 0.0, "g_singleColor5_2");
                        EmitTable(w, 6, 2, 0.0, "g_singleColor6_2");
                        EmitTable(w, 5, 3, 0.03, "g_singleColor5_3_p");
                        EmitTable(w, 6, 3, 0.03, "g_singleColor6_3_p");
                        EmitTable(w, 5, 2, 0.03, "g_singleColor5_2_p");
                        EmitTable(w, 6, 2, 0.03, "g_singleColor6_2_p");
                    }

                    w.WriteLine("}}}");
                }
            }

            MakeETC2AlphaRoundingTables("ConvectionKernels_ETC2_Rounding.h");
            MakeFakeBT709RoundingTables("ConvectionKernels_FakeBT709_Rounding.h");
        }
    }
}
