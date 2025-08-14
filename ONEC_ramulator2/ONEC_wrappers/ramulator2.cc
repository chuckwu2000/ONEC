#include "ramulator2.hh"

#include "base/base.h"
#include "base/request.h"
#include "base/config.h"
#include "frontend/frontend.h"
#include "memory_system/memory_system.h"

namespace OEMSim {
	void Ramulator2::init() {
		num_reads = 0;
		num_writes = 0;
		total_dram_access_ps = 0;
		YAML::Node config = Ramulator::Config::parse_config_file(config_path, {});
		ramulator2_frontend = Ramulator::Factory::create_frontend(config);
		ramulator2_memorysystem = Ramulator::Factory::create_memory_system(config);
		ramulator2_frontend->connect_memory_system(ramulator2_memorysystem);
		ramulator2_memorysystem->connect_frontend(ramulator2_frontend);
	}

	bool Ramulator2::full() const {
		return request_queue.size() >= 256;
	}

	void Ramulator2::push(mem_fetch* mf) {
		request_queue.push(mf);
	}

	void Ramulator2::finish() {
		ramulator2_frontend->finalize();
		ramulator2_memorysystem->finalize();
	}

	void Ramulator2::cycle()
	{
		if(!request_queue.empty())
		{
			mem_fetch* mf = request_queue.front();

			auto callback = [this, mf](Ramulator::Request& req)
			{
				if(req.type_id == Ramulator::Request::Type::Read) {
					num_reads += dram_one_time_request_size;
				} else {
					num_writes += dram_one_time_request_size;
				}
				remain_request -= dram_one_time_request_size;
			};
			
			bool success = ramulator2_frontend->receive_external_requests(mf->is_write(), mf->addr, 0, callback);
			if(success)
			{
				request_queue.pop();
			}
		}
		ramulator2_memorysystem->tick();
		total_dram_access_ps += ramulator2_memorysystem->get_tCK();
	}
}
