#ifndef __RAMULATOR2_HH__
#define __RAMULATOR2_HH__

#include <deque>
#include <functional>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>

namespace Ramulator {
	class IFrontEnd;
	class IMemorySystem;
}

namespace OEMSim {
	struct mem_fetch {
		uint64_t addr;
		bool write;
		bool is_write() const { return write; }
	};

	class Ramulator2 {
		public:
			Ramulator2(std::string ramulator2_config, long long int &size, int dram_one_time_request_size)
			: config_path(ramulator2_config), remain_request(size), dram_one_time_request_size(dram_one_time_request_size)
			{
				init();
			}
			~Ramulator2() {}
			void init();
			bool full() const;
			void finish();
			void push(class mem_fetch* mf);
			void cycle();
			void get_stats(int &num_reads, int &num_writes, double &total_dram_access_ps)
			{
				num_reads = this->num_reads;
				num_writes = this->num_writes;
				total_dram_access_ps = this->total_dram_access_ps;
			}
		private:
			std::string config_path;
			long long int &remain_request;
			int dram_one_time_request_size;
			std::queue<mem_fetch*> request_queue;
			Ramulator::IFrontEnd *ramulator2_frontend;
			Ramulator::IMemorySystem *ramulator2_memorysystem;
			int num_reads;
			int num_writes;
			double total_dram_access_ps;
	};
} // namespace OEMSim
#endif // __RAMULATOR2_HH__