#include <iostream>
#include <string>
#include <memory>
#include <cmath>
#include "ramulator2.hh"

using namespace std;

int main(int argc, char* argv[])
{
	if(argc != 5)
	{
		cout<<"Invalid number of args when using Ramulator2 !!"<<endl;
	}
	string dram_config_path = argv[1];
	uint64_t src_addr = stoll(argv[2]);
	long long int size = stoll(argv[3]);
	int read_write = stoi(argv[4]);
	int dram_one_time_request_size = 64;
	int max_transaction = ceil(float(size) / dram_one_time_request_size);
	int transaction = 0;
	unique_ptr<OEMSim::Ramulator2> dram;
	dram = make_unique<OEMSim::Ramulator2>(dram_config_path, size, dram_one_time_request_size);
	while(size > 0)
	{
		if(!dram->full())
		{
			if(transaction < max_transaction)
			{
				OEMSim::mem_fetch* mf = new OEMSim::mem_fetch();
				mf->addr = src_addr + transaction * dram_one_time_request_size;
				mf->write = read_write;
				transaction++;
				dram->push(mf);
			}
		}
		dram->cycle();
	}
	int num_reads, num_writes;
	double total_dram_access_ps, power;
	dram->get_stats(num_reads, num_writes, total_dram_access_ps);
	cout<<num_reads<<" "<<num_writes<<" "<<total_dram_access_ps<<endl;
	dram->finish();
	return 0;
}