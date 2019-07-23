#pragma once

#pragma warning(disable:4996)
#include <array>
#include <random>
#include <algorithm>
#include <string>
using namespace std;

class game_node
{
public:
	void start() noexcept
	{
		mciSendString("play g1 from 0", NULL, 0, NULL);
		LOGFONT f;
		gettextstyle(&f);						
		f.lfHeight = 24;						
		_tcscpy(f.lfFaceName, _T("黑体"));		
		f.lfQuality = ANTIALIASED_QUALITY;		
		settextstyle(&f);						

		level = 1;
		global::gaming() = true;
		score_helper::read_score();
		score_table.fill(0);
		timer = 0;
	}
	void end() noexcept
	{
		global::gaming() = false;
		
		score_helper::write_score();
	}
	void execute() noexcept
	{
		// 核心逻辑部分
		// GAME ->
		//           ESC  回到STOP
		//           输了  回到MAIN
		constexpr uint8_t GOTO_STOP = 27;

		while (global::now_key() != GOTO_STOP)
		{// Gaming zone.

			static uint32_t pauser = 0;

			if (lose())
			{
				end();
				global::state() = global::states::MAIN;
				return;
			}

			if (timer % (game_property::time_slice / game_property::draw_pause) == 0)
			{
				if ((pauser++) % 6 == 0)
				{
					gen_arr(); // 更新map
					draw();
				}
				else if (pauser % 6 == 5)
				{
					table.fill(obj_type::EMPTY);
					draw();
				}
			}

			if (game_property::valid_map[global::now_key()] != game_property::nullkey)
			{
				// 打击！
				auto s = score_gain(table[game_property::valid_map[global::now_key()]]);
				if (s > 0)
					global::music_id() = ADD;
				else if(s < 0)
					global::music_id() = SUB;
				score_table[level-1] += s;
				table[game_property::valid_map[global::now_key()]] = EMPTY;
				for (size_t i = 0; i < 4; i++)
					score_helper::update(i + 1, max(0, score_table[i]));
				draw();
			}			
			// For each epoch;
			SLEEP(game_property::draw_pause);
			++timer;
		}
		global::state() = global::states::STOP;
	}
private:
	uint16_t level = 1;
	size_t timer   = 0;
	array<obj_type, 9> table;
	array<int, 4> score_table;
private://private:
	size_t to_millisec() noexcept
	{
		return timer * game_property::draw_pause;
	}
	void draw() noexcept
	{
		static bool flap = true;
		putimage(0, 0, &im_manager::game_n_base);
		for (size_t i = 0; i < table.size(); i++)
		{
			if (table[i] != obj_type::EMPTY)
				putimage(game_property::coordinates[i].x, game_property::coordinates[i].y, &im_manager::icons[table[i]-1][flap=!flap]);
		}
		static RECT time_r = { 90, 120, 150, 380 };
		drawtext(_T(to_string(timer*game_property::draw_pause/1000).c_str()), &time_r, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
		static RECT level_r = { 125, 185, 150, 450 };
		drawtext(_T(to_string(level).c_str()), &level_r, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
		static RECT mscore_r = { 155, 300, 200, 460 };
		drawtext(_T(to_string(score_helper::max_score(level)).c_str()), &mscore_r, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
		static RECT score_r = { 155, 425, 200, 470 };
		drawtext(_T(to_string(score_table[level - 1]).c_str()), &score_r, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
	}
	void gen_arr() noexcept
	{
		static random_device rd;
		static default_random_engine e(rd());
		static bernoulli_distribution d(0.333333333333333); // let 9 * p = 3
		uniform_int_distribution<int> ud(1, level - (level > 1));
		for (auto & x : table)
			x = (d(e)) ? (obj_type)ud(e) : obj_type::EMPTY;
	}
	bool lose() noexcept
	{
		switch (level)
		{
		case 1:
			if (score_table[0] >= 60)
			{
				++level;
				mciSendString("play g2 from 0 wait", NULL, 0, NULL);
			}
			return false;
			break;
		case 2:
			if (score_table[1] >= 60)
			{
				++level;
				mciSendString("play g3 from 0 wait", NULL, 0, NULL);
			}
			return score_table[1] < 0;
			break;
		case 3:
			if (score_table[2] >= 60)
			{
				mciSendString("play g4 from 0 wait", NULL, 0, NULL);
				++level;
			}
			return score_table[2] < 0;
			break;
		case 4:
			return score_table[3] < 0;
			break;
		}
	}
	int score_gain(obj_type t) noexcept
	{
		constexpr array<array<int, 4>, 4> scores{ {
													{0,  -1, -1, -1},
													{1,  1,  1,  1 },
													{-2, -2, -2, -2},
													{10, 10, 10, 10}  } };
		return scores[t][level - 1];
	}
};