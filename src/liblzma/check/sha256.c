// SPDX-License-Identifier: 0BSD

///////////////////////////////////////////////////////////////////////////////
//
/// \file       sha256.c
/// \brief      SHA-256
//
//  The C code is based on the public domain SHA-256 code found from
//  Crypto++ Library 5.5.1 released in 2007: https://www.cryptopp.com/
//  A few minor tweaks have been made in liblzma.
//
//  The x86 intrinsics code is based on Intel documentation:
//  https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sha-extensions.html
//
//  Authors:    Wei Dai
//              Lasse Collin
//
///////////////////////////////////////////////////////////////////////////////

#include "check.h"

// If defined, generic C version is built. This will be undefined when
// compiler flags allow unconditional use of the SHA-256 instructions.
#define SHA256_GENERIC 1

#ifdef HAVE_SHA256_X86
#	include <immintrin.h>
#	if defined(__SHA__)
		// No need for a runtime check or the generic C code.
#		undef SHA256_GENERIC
#		define is_arch_extension_supported() true
#	elif defined(_MSC_VER)
#		include <intrin.h>
#	else
#		include <cpuid.h>
#	endif
#endif


#if defined(HAVE_SHA256_X86) && defined(SHA256_GENERIC)
static bool
is_arch_extension_supported(void)
{
	unsigned int r[4]; // eax, ebx, ecx, edx

#if defined(_MSC_VER) || !defined(HAVE_CPUID_H)
	__cpuid((int *)r, 0);
	if (r[0] < 7)
		return false;

	__cpuidex((int *)r, 7, 0);
#else
	// Old GCC lacks __get_cpuid_count(), so use two steps:
	if (__get_cpuid_max(0, NULL) < 7)
		return false;

	__cpuid_count(7, 0, r[0], r[1], r[2], r[3]);
#endif

	// Bit 29 in ebx indicates support for SHA-1 and SHA-256 extensions.
	// If it is set, we assume that the other required extensions like
	// SSSE3 are also supported (we need PSHUFB and PALIGNR).
	return (r[1] & (1 << 29)) != 0;
}
#endif


// Align the constants for SSE2 instructions.
alignas(16)
static const uint32_t SHA256_K[64] = {
	0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5,
	0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
	0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3,
	0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
	0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC,
	0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
	0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7,
	0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
	0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13,
	0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
	0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3,
	0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
	0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5,
	0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
	0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208,
	0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
};


#ifdef SHA256_GENERIC
// Rotate a uint32_t. GCC can optimize this to a rotate instruction
// at least on x86.
static inline uint32_t
rotr_32(uint32_t num, unsigned amount)
{
	return (num >> amount) | (num << (32 - amount));
}

#define blk0(i) (W[i] = conv32be(data[i]))
#define blk2(i) (W[i & 15] += s1(W[(i - 2) & 15]) + W[(i - 7) & 15] \
		+ s0(W[(i - 15) & 15]))

#define Ch(x, y, z) (z ^ (x & (y ^ z)))
#define Maj(x, y, z) ((x & (y ^ z)) + (y & z))

#define a(i) T[(0 - i) & 7]
#define b(i) T[(1 - i) & 7]
#define c(i) T[(2 - i) & 7]
#define d(i) T[(3 - i) & 7]
#define e(i) T[(4 - i) & 7]
#define f(i) T[(5 - i) & 7]
#define g(i) T[(6 - i) & 7]
#define h(i) T[(7 - i) & 7]

#define R(i, j, blk) \
	h(i) += S1(e(i)) + Ch(e(i), f(i), g(i)) + SHA256_K[i + j] + blk; \
	d(i) += h(i); \
	h(i) += S0(a(i)) + Maj(a(i), b(i), c(i))
#define R0(i) R(i, 0, blk0(i))
#define R2(i) R(i, j, blk2(i))

#define S0(x) rotr_32(x ^ rotr_32(x ^ rotr_32(x, 9), 11), 2)
#define S1(x) rotr_32(x ^ rotr_32(x ^ rotr_32(x, 14), 5), 6)
#define s0(x) (rotr_32(x ^ rotr_32(x, 11), 7) ^ (x >> 3))
#define s1(x) (rotr_32(x ^ rotr_32(x, 2), 17) ^ (x >> 10))


static void
transform(uint32_t state[8], const uint32_t data[16])
{
	uint32_t W[16];
	uint32_t T[8];

	// Copy state[] to working vars.
	memcpy(T, state, sizeof(T));

	// The first 16 operations unrolled
	R0( 0); R0( 1); R0( 2); R0( 3);
	R0( 4); R0( 5); R0( 6); R0( 7);
	R0( 8); R0( 9); R0(10); R0(11);
	R0(12); R0(13); R0(14); R0(15);

	// The remaining 48 operations partially unrolled
	for (unsigned int j = 16; j < 64; j += 16) {
		R2( 0); R2( 1); R2( 2); R2( 3);
		R2( 4); R2( 5); R2( 6); R2( 7);
		R2( 8); R2( 9); R2(10); R2(11);
		R2(12); R2(13); R2(14); R2(15);
	}

	// Add the working vars back into state[].
	state[0] += a(0);
	state[1] += b(0);
	state[2] += c(0);
	state[3] += d(0);
	state[4] += e(0);
	state[5] += f(0);
	state[6] += g(0);
	state[7] += h(0);
}
#endif // SHA256_GENERIC


#ifdef HAVE_SHA256_X86
#if (defined(__GNUC__) || defined(__clang__)) && !defined(__EDG__)
__attribute__((__target__("ssse3,sha")))
#endif
static void
transform_arch_optimized(__m128i state[2], const __m128i data[4])
{
	const __m128i byteswap_mask = _mm_set_epi32(
			0x0C0D0E0F, 0x08090A0B, 0x04050607, 0x00010203);

	__m128i state0 = state[0];
	__m128i state1 = state[1];

	__m128i msg;
	__m128i msgtmp0;
	__m128i msgtmp1;
	__m128i msgtmp2;
	__m128i msgtmp3;

	// Rounds 0-3
	msgtmp0 = _mm_shuffle_epi8(data[0], byteswap_mask);
	msg = _mm_add_epi32(msgtmp0,
			_mm_load_si128((const __m128i *)(SHA256_K + 0)));
	state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
	msg = _mm_shuffle_epi32(msg, 0x0E);
	state0 = _mm_sha256rnds2_epu32(state0, state1, msg);

	// Rounds 4-7
	msgtmp1 = _mm_shuffle_epi8(data[1], byteswap_mask);
	msg = _mm_add_epi32(msgtmp1,
			_mm_load_si128((const __m128i *)(SHA256_K + 4)));
	state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
	msg = _mm_shuffle_epi32(msg, 0x0E);
	state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
	msgtmp0 = _mm_sha256msg1_epu32(msgtmp0, msgtmp1);

	// Rounds 8-11
	msgtmp2 = _mm_shuffle_epi8(data[2], byteswap_mask);
	msg = _mm_add_epi32(msgtmp2,
			_mm_load_si128((const __m128i *)(SHA256_K + 8)));
	state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
	msg = _mm_shuffle_epi32(msg, 0x0E);
	state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
	msgtmp1 = _mm_sha256msg1_epu32(msgtmp1, msgtmp2);

	// Prepare for round 12
	msgtmp3 = _mm_shuffle_epi8(data[3], byteswap_mask);

	for (size_t j = 12; ; j += 16) {
		// Rounds 12-15, 28-31, 44-47
		msg = _mm_add_epi32(msgtmp3,
			_mm_load_si128((const __m128i *)(SHA256_K + j)));
		state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
		msgtmp0 = _mm_add_epi32(msgtmp0,
			_mm_alignr_epi8(msgtmp3, msgtmp2, 4));
		msgtmp0 = _mm_sha256msg2_epu32(msgtmp0, msgtmp3);
		msg = _mm_shuffle_epi32(msg, 0x0E);
		state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
		msgtmp2 = _mm_sha256msg1_epu32(msgtmp2, msgtmp3);

		// Rounds 16-19, 32-35, 48-51
		msg = _mm_add_epi32(msgtmp0,
			_mm_load_si128((const __m128i *)(SHA256_K + j + 4)));
		state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
		msgtmp1 = _mm_add_epi32(msgtmp1,
			_mm_alignr_epi8(msgtmp0, msgtmp3, 4));
		msgtmp1 = _mm_sha256msg2_epu32(msgtmp1, msgtmp0);
		msg = _mm_shuffle_epi32(msg, 0x0E);
		state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
		msgtmp3 = _mm_sha256msg1_epu32(msgtmp3, msgtmp0);

		// Rounds 20-23, 36-39, 52-55
		msg = _mm_add_epi32(msgtmp1,
			_mm_load_si128((const __m128i *)(SHA256_K + j + 8)));
		state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
		msgtmp2 = _mm_add_epi32(msgtmp2,
			_mm_alignr_epi8(msgtmp1, msgtmp0, 4));
		msgtmp2 = _mm_sha256msg2_epu32(msgtmp2, msgtmp1);
		msg = _mm_shuffle_epi32(msg, 0x0E);
		state0 = _mm_sha256rnds2_epu32(state0, state1, msg);

		if (j == 52 - 8)
			break;

		msgtmp0 = _mm_sha256msg1_epu32(msgtmp0, msgtmp1);

		// Rounds 24-27, 40-43
		msg = _mm_add_epi32(msgtmp2,
			_mm_load_si128((const __m128i *)(SHA256_K + j + 12)));
		state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
		msgtmp3 = _mm_add_epi32(msgtmp3,
			_mm_alignr_epi8(msgtmp2, msgtmp1, 4));
		msgtmp3 = _mm_sha256msg2_epu32(msgtmp3, msgtmp2);
		msg = _mm_shuffle_epi32(msg, 0x0E);
		state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
		msgtmp1 = _mm_sha256msg1_epu32(msgtmp1, msgtmp2);
	}

	// Rounds 56-59
	msg = _mm_add_epi32(msgtmp2,
			_mm_load_si128((const __m128i *)(SHA256_K + 56)));
	state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
	msgtmp3 = _mm_add_epi32(msgtmp3, _mm_alignr_epi8(msgtmp2, msgtmp1, 4));
	msgtmp3 = _mm_sha256msg2_epu32(msgtmp3, msgtmp2);
	msg = _mm_shuffle_epi32(msg, 0x0E);
	state0 = _mm_sha256rnds2_epu32(state0, state1, msg);

	// Rounds 60-63
	msg = _mm_add_epi32(msgtmp3,
			_mm_load_si128((const __m128i *)(SHA256_K + 60)));
	state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
	msg = _mm_shuffle_epi32(msg, 0x0E);
	state0 = _mm_sha256rnds2_epu32(state0, state1, msg);

	// Add the working vars back into state[].
	state[0] = _mm_add_epi32(state[0], state0);
	state[1] = _mm_add_epi32(state[1], state1);
}
#endif


static void
process(lzma_check_state *check)
{
#ifdef HAVE_SHA256_X86
	if (check->state.sha256.use_arch_extension) {
		transform_arch_optimized((__m128i *)check->state.sha256.state,
				check->buffer.m128);
	} else
#endif
	{
#ifdef SHA256_GENERIC
		transform(check->state.sha256.state, check->buffer.u32);
#endif
	}
	return;
}


extern void
lzma_sha256_init(lzma_check_state *check)
{
	static const uint32_t s[8] = {
		0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
		0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
	};

#ifdef HAVE_SHA256_X86
	check->state.sha256.use_arch_extension = is_arch_extension_supported();
	if (check->state.sha256.use_arch_extension) {
		// In s[], there are 8 32-bit values in the order ABCDEFGH.
		// However, the sha256rnds2 instruction takes the state as two
		// xmm operands where the little endian order is ABEF and CDGH.
		//
		// ABEF:
		check->state.sha256.state[0] = s[5];
		check->state.sha256.state[1] = s[4];
		check->state.sha256.state[2] = s[1];
		check->state.sha256.state[3] = s[0];
		// CDGH:
		check->state.sha256.state[4] = s[7];
		check->state.sha256.state[5] = s[6];
		check->state.sha256.state[6] = s[3];
		check->state.sha256.state[7] = s[2];
	} else
#endif
	{
#ifdef SHA256_GENERIC
		memcpy(check->state.sha256.state, s, sizeof(s));
#endif
	}

	check->state.sha256.size = 0;
	return;
}


extern void
lzma_sha256_update(const uint8_t *buf, size_t size, lzma_check_state *check)
{
	// Copy the input data into a properly aligned temporary buffer.
	// This way we can be called with arbitrarily sized buffers
	// (no need to be multiple of 64 bytes), and the code works also
	// on architectures that don't allow unaligned memory access.
	while (size > 0) {
		const size_t copy_start = check->state.sha256.size & 0x3F;
		size_t copy_size = 64 - copy_start;
		if (copy_size > size)
			copy_size = size;

		memcpy(check->buffer.u8 + copy_start, buf, copy_size);

		buf += copy_size;
		size -= copy_size;
		check->state.sha256.size += copy_size;

		if ((check->state.sha256.size & 0x3F) == 0)
			process(check);
	}

	return;
}


extern void
lzma_sha256_finish(lzma_check_state *check)
{
	// Add padding as described in RFC 3174 (it describes SHA-1 but
	// the same padding style is used for SHA-256 too).
	size_t pos = check->state.sha256.size & 0x3F;
	check->buffer.u8[pos++] = 0x80;

	while (pos != 64 - 8) {
		if (pos == 64) {
			process(check);
			pos = 0;
		}

		check->buffer.u8[pos++] = 0x00;
	}

	// Convert the message size from bytes to bits.
	check->state.sha256.size *= 8;

	check->buffer.u64[(64 - 8) / 8] = conv64be(check->state.sha256.size);

	process(check);

#ifdef HAVE_SHA256_X86
	if (check->state.sha256.use_arch_extension) {
		const uint32_t tmp[8] = {
			// Convert from ABEF CDGH to ABCDEFGH, reversing
			// what was described in lzma_sha256_init().
			check->state.sha256.state[3],
			check->state.sha256.state[2],
			check->state.sha256.state[7],
			check->state.sha256.state[6],
			check->state.sha256.state[1],
			check->state.sha256.state[0],
			check->state.sha256.state[5],
			check->state.sha256.state[4],
		};
		memcpy(check->state.sha256.state, tmp, sizeof(tmp));
	}
#endif

	for (size_t i = 0; i < 8; ++i)
		check->buffer.u32[i] = conv32be(check->state.sha256.state[i]);

	return;
}
