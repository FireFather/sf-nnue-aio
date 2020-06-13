﻿#ifndef __HALF_FLOAT_H__
#define __HALF_FLOAT_H__

// Half Float Library by yaneurao
// (16-bit float)

// Floating point operation by 16bit type
// Assume that the float type code generated by the compiler is in IEEE 754 format and use it.

#include <iostream>

#include "../types.h"

namespace HalfFloat
{
	// IEEE 754 float 32 format is:
	// sign(1bit) + exponent(8bits) + fraction(23bits) = 32bits
	//
	// Our float16 format is:
	// sign(1bit) + exponent(5bits) + fraction(10bits) = 16bits
	union float32_converter
	{
		int32_t n;
		float f;
	};


	// 16-bit float
	struct float16
	{
		// --- constructors

		float16() {}
		float16(int16_t n) { from_float(static_cast<float>(n)); }
		float16(int32_t n) { from_float(static_cast<float>(n)); }
		float16(float n) { from_float(n); }
		float16(double n) { from_float(static_cast<float>(n)); }

		// build from a float
		void from_float(float f) { *this = to_float16(f); }

		// --- implicit converters

		operator int32_t() const { return static_cast<int32_t>(to_float(*this)); }
		operator float() const { return to_float(*this); }
		operator double() const { return static_cast<double>(to_float(*this)); }

		// --- operators

		float16 operator += (float16 rhs) { from_float(to_float(*this) + to_float(rhs)); return *this; }
		float16 operator -= (float16 rhs) { from_float(to_float(*this) - to_float(rhs)); return *this; }
		float16 operator *= (float16 rhs) { from_float(to_float(*this) * to_float(rhs)); return *this; }
		float16 operator /= (float16 rhs) { from_float(to_float(*this) / to_float(rhs)); return *this; }
		float16 operator + (float16 rhs) const { return float16(*this) += rhs; }
		float16 operator - (float16 rhs) const { return float16(*this) -= rhs; }
		float16 operator * (float16 rhs) const { return float16(*this) *= rhs; }
		float16 operator / (float16 rhs) const { return float16(*this) /= rhs; }
		float16 operator - () const { return float16(-to_float(*this)); }
		bool operator == (float16 rhs) const { return this->v_ == rhs.v_; }
		bool operator != (float16 rhs) const { return !(*this == rhs); }

static void UnitTest() { unit_test(); }

private:

	// --- entity

	uint16_t v_;

	// --- conversion between float and float16

	static float16 to_float16(float f)
	{
		float32_converter c;
		c.f = f;
		uint32_t n = c.n;

		// The sign bit is MSB in common.
		uint16_t sign_bit = n >> 16 & 0x8000;

		// The exponent of IEEE 754's float 32 is biased +127 ,so we change this bias into +15 and limited to 5-bit.
		uint16_t exponent = ((n >> 23) - 127 + 15 & 0x1f) << 10;

		// The fraction is limited to 10-bit.
		uint16_t fraction = n >> 23 - 10 & 0x3ff;

		float16 f_;
		f_.v_ = sign_bit | exponent | fraction;

		return f_;
	}

	static float to_float(float16 v)
	{
		uint32_t sign_bit = (v.v_ & 0x8000) << 16;
		uint32_t exponent = ((v.v_ >> 10 & 0x1f) - 15 + 127 & 0xff) << 23;
		uint32_t fraction = (v.v_ & 0x3ff) << 23 - 10;

		float32_converter c;
		c.n = sign_bit | exponent | fraction;
		return c.f;
	}

	// It is not a unit test, but I confirmed that it can be calculated. I'll fix the code later (maybe).
	static void unit_test()
	{
		float16 a = 1;
		std::cout << static_cast<float>(a) << std::endl;
		float16 b = -118.625;
		std::cout << static_cast<float>(b) << std::endl;
		float16 c = 2.5;
		std::cout << static_cast<float>(c) << std::endl;
		float16 d = a + c;
		std::cout << static_cast<float>(d) << std::endl;

		c *= 1.5;
		std::cout << static_cast<float>(c) << std::endl;

		b /= 3;
		std::cout << static_cast<float>(b) << std::endl;

		float f1 = 1.5;
		a += f1;
		std::cout << static_cast<float>(a) << std::endl;

		a += f1 * static_cast<float>(a);
		std::cout << static_cast<float>(a) << std::endl;
	}

};

}

#endif // __HALF_FLOAT_H__