#pragma once

#define MAP_SZ     50
#define x_shift    200
#define y_shift    0
#define pixel_size 10

#include <cstdint>
#include <deque>
#include <array>
#include <random>
#include <thread>
#include <graphics.h>


/* map = 500 x (200 + 500) */


using namespace std;

enum mode
{
	unknown = -1,
	easy = 0,
	mid = 1,
	hard = 2,
	fly = 3
};

enum direction
{
	north = 1,
	south = -1,
	west = 2,
	east = -2
};

namespace condition
{

	constexpr uint8_t empty = 0;
	constexpr uint8_t normal_fruit = 1;
	constexpr uint8_t good_fruit = 2;
	constexpr uint8_t bad_fruit = 3;
	constexpr uint8_t snake = 4;

	constexpr uint8_t wall = 255;

	inline void condition_set(uint8_t con) noexcept
	{
		/*
		setfillcolor(YELLOW);
		setlinecolor(RED);
		*/
		switch (con)
		{
		case empty:
			/* Do nothing. */
			break;
		case normal_fruit:
			setfillcolor(WHITE);
			setlinecolor(GREEN);
			break;
		case good_fruit:
			setfillcolor(LIGHTCYAN);
			setlinecolor(LIGHTRED);
			break;
		case bad_fruit:
			setfillcolor(GREEN);
			setlinecolor(BLUE);
			break;
		case wall:
			setfillcolor(BLACK);
			setlinecolor(BLACK);
			break;
		case snake:
			setfillcolor(YELLOW);
			setlinecolor(RED);
		default:
			break;
		}
	}
}

class game_map
{
public:
	array<array<uint8_t, MAP_SZ>, MAP_SZ> m_map;
	deque<POINT> snake_list;
	deque<POINT> dead_nodes;
	int fruit_num;
	int empty_num;
	int scores;
	int lifes = 5;
	mode game_mode = mode::unknown;
	direction dir = direction::east;
	direction last_dir = direction::east;
	bool drunk;
private:
	int ad;
public:
	inline game_map() noexcept
	{}
	inline void init(mode m) noexcept
	{
		game_mode = m;
		drunk = false;
		ad = 0;
		scores = 0;
		fruit_num = 0;
		for (size_t i = 0; i < m_map.size(); ++i)
			for (size_t j = 0; j < m_map[i].size(); ++j)
				m_map[i][j] = (i * j == 0 || i == MAP_SZ - 1 || j == MAP_SZ - 1) ? condition::wall : condition::empty;

		if (lifes <= 0 || game_mode != mode::hard)
			lifes = 5;

		if (game_mode == mode::hard)
			for (const auto& p : dead_nodes)
			{
				++fruit_num;
				m_map[p.x][p.y] = condition::normal_fruit;
			}

		if (game_mode == mode::mid)
			for (const auto& p : dead_nodes)
				m_map[p.x][p.y] = condition::wall;
		else
			dead_nodes.clear();

		empty_num = (MAP_SZ - 2) * (MAP_SZ - 2) - dead_nodes.size();

		if (game_mode == mode::fly)
			for (size_t i = 0; i < 15; i++)
			{
				auto tmp = next_point();
				m_map[tmp.x][tmp.y] = condition::bad_fruit;
				tmp = next_point();
				m_map[tmp.x][tmp.y] = condition::good_fruit;
			}


		snake_list.clear();
		snake_list.push_back(next_point());
		m_map[snake_list.front().x][snake_list.front().y] = condition::snake;
		
		int init_num = (game_mode == mode::fly) ? 200 : 5;
		for (size_t i = 0; i < init_num; i++)
		{
			auto p = next_point();
			m_map[p.x][p.y] = condition::normal_fruit;
			++fruit_num;
		}
	}
	inline POINT next_point() noexcept
	{
		static random_device rd;
		static mt19937 engine{ rd() };
		uniform_int_distribution<uint32_t> u(2, empty_num);
		auto check_now = u(engine);
		for (int i = 0; i < MAP_SZ; ++i)
			for (int j = 0; j < MAP_SZ; ++j)
				if (m_map[i][j] == condition::empty && --check_now == 0)
				{
					--empty_num;
					return { i, j };
				}
		return { 0, 0 };
	}
	inline bool snake_next_step(bool pop = true) noexcept
	{
		if (drunk)
			dir = (direction)(-int(dir));

		if (last_dir == -dir)
			dir = last_dir;

		int x = snake_list.back().x + (dir == direction::east) - (dir == direction::west);
		int y = snake_list.back().y + (dir == direction::south) - (dir == direction::north);

		last_dir = dir;
		switch (m_map[x][y])
		{
		case condition::empty:
			snake_proceed(x, y);
			break;
		case condition::normal_fruit:
			scores += 2;
			m_map[x][y] = condition::empty;
			snake_proceed(x, y, false);
			--fruit_num;
			if (fruit_num < 5 )
			{
				auto tmp = next_point();
				m_map[tmp.x][tmp.y] = condition::normal_fruit;
				++fruit_num;
			}
			break;
		case condition::good_fruit:
			m_map[x][y] = condition::empty;
			auto tmp = next_point();
			m_map[tmp.x][tmp.y] = condition::good_fruit;
			drunk = false;
			scores += 10;
			break;
		case condition::bad_fruit:
			auto tmp2 = next_point();
			m_map[tmp2.x][tmp2.y] = condition::bad_fruit;
			m_map[x][y] = condition::empty;
			drunk = true;
			break;
		case condition::wall:
		case condition::snake:
			dead_nodes.insert(dead_nodes.end(), snake_list.begin(), snake_list.end());
			if (game_mode == mode::hard)
				--lifes;
			return false;
		default:
			break;
		}
		return true;
	}
	inline void snake_proceed(int x, int y, bool pop = true) noexcept
	{
		m_map[x][y] = condition::snake;
		snake_list.push_back({ x, y });
		if (pop)
		{
			m_map[snake_list.front().x][snake_list.front().y] = condition::empty;
			snake_list.pop_front();
		}
	}
	inline void draw() noexcept
	{
		BeginBatchDraw();
		setfillcolor(WHITE);
		setlinecolor(WHITE);
		rectangle(x_shift + pixel_size, y_shift + pixel_size, x_shift + (MAP_SZ - 1) * pixel_size, y_shift + (MAP_SZ - 1)  * pixel_size);
		fillrectangle(x_shift + pixel_size, y_shift + pixel_size, x_shift + (MAP_SZ - 1) * pixel_size, y_shift + (MAP_SZ - 1)  * pixel_size);

		for (int i = ad; i < MAP_SZ - ad; ++i)
			for (int j = ad; j < MAP_SZ - ad; ++j)
				if (m_map[i][j] != condition::empty)
				{
					condition::condition_set(m_map[i][j]);
					rectangle(x_shift + i * pixel_size, y_shift + j * pixel_size, x_shift + (i + 1) * pixel_size, y_shift + (1 + j) * pixel_size);
					fillrectangle(x_shift + i * pixel_size, y_shift + j * pixel_size, x_shift + (i + 1) * pixel_size, y_shift + (1 + j) * pixel_size);
				}
		EndBatchDraw();
		ad = 1;
	}
};