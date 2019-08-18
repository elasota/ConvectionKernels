#pragma once

#include <stdint.h>

typedef struct KtxHeader_s
{
	enum class InternalFormat
	{
		ETC1_RGB8 = 0x8D64,
		ETC1_ALPHA8 = ETC1_RGB8,
		//
		ETC2_R11 = 0x9270,
		ETC2_SIGNED_R11 = 0x9271,
		ETC2_RG11 = 0x9272,
		ETC2_SIGNED_RG11 = 0x9273,
		ETC2_RGB8 = 0x9274,
		ETC2_SRGB8 = 0x9275,
		ETC2_RGB8A1 = 0x9276,
		ETC2_SRGB8_PUNCHTHROUGH_ALPHA1 = 0x9277,
		ETC2_RGBA8 = 0x9278
	};

	enum class BaseInternalFormat
	{
		ETC2_R11 = 0x1903,
		ETC2_RG11 = 0x8227,
		ETC1_RGB8 = 0x1907,
		ETC1_ALPHA8 = ETC1_RGB8,
		//
		ETC2_RGB8 = 0x1907,
		ETC2_RGB8A1 = 0x1908,
		ETC2_RGBA8 = 0x1908,
	};

	uint8_t identifier[12];
	uint32_t endianness;
	uint32_t glType;
	uint32_t glTypeSize;
	uint32_t glFormat;
	uint32_t glInternalFormat;
	uint32_t glBaseInternalFormat;
	uint32_t pixelWidth;
	uint32_t pixelHeight;
	uint32_t pixelDepth;
	uint32_t numberOfArrayElements;
	uint32_t numberOfFaces;
	uint32_t numberOfMipmapLevels;
	uint32_t bytesOfKeyValueData;
} KtxHeader_t;
