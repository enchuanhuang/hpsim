#include <iostream>
#include "timer.h"
#include "MsgLog.h"

void StartTimer(cudaEvent_t* r_start, cudaEvent_t* r_stop)
{
  cudaEventCreate(r_start);
  cudaEventCreate(r_stop);
  cudaEventRecord(*r_start, 0);
}
void StopTimer(cudaEvent_t* r_start, cudaEvent_t* r_stop, std::string r_msg)
{
  cudaEventRecord(*r_stop, 0);
  cudaEventSynchronize(*r_stop);
  float elapst;
  cudaEventElapsedTime(&elapst, *r_start, *r_stop);
  MsgInfo(MsgLog::Form("%s %.4f [sec]\n", r_msg.c_str(), elapst/1000));
  //std::cout << r_msg << " time: " << elapst / 1000 << " [sec]" << std::endl;
  cudaEventDestroy(*r_start);
  cudaEventDestroy(*r_stop);
}
