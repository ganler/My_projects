#pragma once

#include <iostream>
#include <cstdint>

template <typename _Tp = int> class array
{
private:
	using value_type = _Tp;
	value_type* m_data = nullptr; // [m_channels] -> m_rows -> m_cols
	std::size_t m_cols = 0;
	std::size_t m_rows = 1;
	std::size_t m_channels = 1;
private:
	void clear() noexcept
	{
		if (m_data != nullptr)
			delete[] m_data;
	}
public:
	array(value_type* src, std::size_t r, std::size_t c, std::size_t cn = 1)
		: m_data(src), m_cols(c), m_rows(r), m_channels(cn) {}
	array(std::size_t r, std::size_t c, std::size_t cn = 1)
		: m_cols(c), m_rows(r), m_channels(cn), m_data(new value_type[c*r*cn]) {}
	~array() noexcept { clear(); }
	array(const array& arr)
	{
		clear();
		if (arr.m_data != nullptr)
		{
			m_data = new value_type[arr.size()];
			m_cols = arr.m_cols;
			m_rows = arr.m_rows;
			m_channels = arr.m_channels;
			std::memcpy(m_data, arr.m_data, sizeof(value_type) * arr.size());
		}
	}
	array(array&& arr) noexcept
	{
		clear();
		if (arr.m_data != nullptr)
		{
			std::swap(arr.m_data, m_data);
			m_cols = arr.m_cols;
			m_rows = arr.m_rows;
			m_channels = arr.m_channels;
		}
	}
	array& operator=(array arr) noexcept
	{
		std::swap(m_data, arr.m_data);
		m_cols = arr.m_cols;
		m_rows = arr.m_rows;
		m_channels = arr.m_channels;
		return *this;
	}
public:
	inline value_type& at(std::size_t r, std::size_t c, std::size_t cn = 1)
	{
		return m_data[m_channels*m_cols*r + m_channels * c + cn];
	}
	inline value_type at(std::size_t r, std::size_t c, std::size_t cn = 1) const
	{
		return m_data[m_channels*m_cols*r + m_channels * c + cn];
	}
	inline std::size_t size() const noexcept
	{
		return m_cols * m_rows*m_channels;
	}
	inline std::size_t rows() const noexcept
	{
		return m_rows;
	}
	inline std::size_t cols() const noexcept
	{
		return m_cols;
	}
	inline std::size_t channels() const noexcept
	{
		return m_channels;
	}
	void reshape(std::size_t r, std::size_t c, std::size_t cn = 1)
	{
		m_rows = r;
		m_cols = c;
		m_channels = cn;
	}
public:
	array operator + (array rhs) const noexcept
	{
		for (std::size_t i = 0; i < size(); ++i)
			rhs.m_data[i] += m_data[i];
		return rhs;
	}
	array operator - (array rhs) const noexcept
	{
		for (std::size_t i = 0; i < size(); ++i)
			rhs.m_data[i] -= m_data[i];
		return rhs;
	}
	array operator * (array rhs) const noexcept
	{
		for (std::size_t i = 0; i < size(); ++i)
			rhs.m_data[i] *= m_data[i];
		return rhs;
	}
	array operator / (array rhs) const noexcept
	{
		for (std::size_t i = 0; i < size(); ++i)
			rhs.m_data[i] /= m_data[i];
		return rhs;
	}
	value_type*& data()
	{
		return m_data;
	}
	const value_type*& data() const
	{
		return m_data;
	}
	decltype(auto) to_ascii() noexcept
	{
		constexpr size_t tms = 2;
		constexpr uint16_t SCALE_SZ = 15;
		constexpr uint16_t r_ = static_cast<uint16_t>(0.299 * (1 << SCALE_SZ) + 0.5);
		constexpr uint16_t g_ = static_cast<uint16_t>(0.587 * (1 << SCALE_SZ) + 0.5);
		constexpr uint16_t b_ = static_cast<uint16_t>(0.114 * (1 << SCALE_SZ) + 0.5);
		constexpr int8_t ascii_code[] = {'#','M','$','N','$','%','?','>','+','=','!',';',':','-',',', '.'};
		auto src = m_data;

		for (int i = 0; i < m_cols * m_rows; ++i)
		{
			std::size_t base = i * channels();
			src[i*tms] = ascii_code[static_cast<uint8_t>(
				(r_ * src[base] + g_ * src[base + 1] + b_ * src[base + 2] + (1 << (SCALE_SZ - 1))) >> SCALE_SZ
				) >> 4];
			src[i*tms + 1] = src[i*tms];
		}
	}
	friend std::ostream& operator << (std::ostream& os, const array& arr)
	{
		const auto line_width = arr.m_cols * arr.m_channels;
		for (int i = 0; i < arr.size(); ++i)
		{
			if (i > 0 && i % line_width == 0)
				os << '\n';
			os << arr.data()[i] << ' ';
		}
		return os << '\n';
	}
	friend std::istream& operator >> (std::istream& is, array& arr)
	{
		for (int i = 0; i < arr.size(); ++i)
			is >> arr.data()[i];
		return is;
	}
};

using Array = array<uint8_t>;
