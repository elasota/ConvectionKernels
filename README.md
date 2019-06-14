# ConvectionKernels
These are the stand-alone texture compression kernels for Convection Texture Tools (CVTT), you can embed these in other applications.
https://github.com/elasota/cvtt

The CVTT codecs are designed to get very high quality at good speed by leveraging effective heuristics and a SPMD-style design that makes heavy use of SIMD ops and 16-bit math.

Compressed texture format support:
 * BC1 (DXT1): Complete
 * BC2 (DXT3): Complete
 * BC3 (DXT5): Complete
 * BC4: Complete
 * BC5: Complete
 * BC6H: Experimental
 * BC7: Complete
 * ETC1: Complete
 * ETC2 RGB: Complete
 * ETC2 RGBA: Complete
 * ETC2 with punchthrough alpha: Not supported
 * 11-bit EAC: Not supported
 * PVRTC: Not supported
 * ASTC: Not supported


# Basic usage

Include "ConvectionKernels.h"

Depending on the input format, blocks should be pre-packed into one of the PixelBlock structures: PixelBlockU8 for unsigned LDR formats (BC1, BC2, BC3, BC7, BC4U, BC5U), PixelBlockS8 for signed LDR formats (BC4S, BC5S), and PixelBlockF16 for HDR formats (BC6H).  The block pixel order is left-to-right, top-to-bottom, and the channel order is red, green, blue, alpha.

BC6H floats are stored as int16_t in the pixel block structure, which should be bit-cast from the 16-bit float input.  Converting other float precisions to 16-bit is outside of the scope of the kernels.

Create an Options structure and fill it out:
  * flags: A bitwise OR mask of one of cvtt::Flags, which enable or disable various features.
  * threshold: The alpha threshold for encoding BC1 with alpha test.  Any alpha value lower than than the threshold will use transparent alpha.
  * redWeight: Red channel relative importance
  * blueWeight: Blue channel relative importance
  * alphaWeight: Alpha channel relative importance
  * seedPoints: Number of seed points to try, from 1 to 4.  Higher values improve quality.

Once you've done that, call the corresponding encode function to digest the input blocks and emit output blocks.

**VERY IMPORTANT**: The encode functions must be given a list of cvtt::NumParallelBlocks blocks, and will emit cvtt::NumParallelBlocks output blocks.  If you want to encode fewer blocks, then you must pad the input structure with unused block data, and the output buffer must still contain enough space.

# ETC compression

The ETC encoders require significantly more temporary data storage than the other encoders, so the storage must be allocated before using the encoders.

To allocate the temporary data:
  * Create an allocation function compatible with cvtt::Kernels::allocFunc_t, which accepts a context pointer and byte size and returns a buffer of at least that size.  The returned buffer must be byte-aligned for SIMD usage (i.e. 16 byte alignment on Intel).
  * Use the AllocETC1Data or AllocETC2Data functions, pass the allocation function and a context pointer, which will be passed back to the allocation function.

To release the temporary data:
  * Create a free function compatible with cvtt::Kernels::freeFunc_t, which accepts a context pointer, a pointer to the buffer allocated by the allocation func, and the original size.
  * Use the ReleaseETC1Data or ReleaseETC2Data functions, pass the original compression data structure returned by the allocation function, and the free function.

Once allocated, the compression data can be reused over multiple calls to the encode functions, and depending on architecture, can usually be used by a different thread than the one that allocated it, as long as multiple encode functions are not using it at once.
