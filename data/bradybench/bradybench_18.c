#pragma ACCEL kernel

void top(float input[32][128],float gamma[128],float beta[128],float output[32][128])
{
  float diff;
  float mean[128] = {(0.0f)};
  float variance[128] = {(0.0f)};
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (int f = 0; f < 128; f++) {
    for (int b = 0; b < 32; b++) {
      mean[f] += input[b][f];
    }
    mean[f] /= ((float )32);
  }
  
#pragma ACCEL PIPELINE auto{__PIPE__L1}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
  for (int f = 0; f < 128; f++) {
    for (int b = 0; b < 32; b++) {
      diff = input[b][f] - mean[f];
      variance[f] += diff * diff;
    }
    variance[f] /= ((float )32);
  }
  
#pragma ACCEL PIPELINE auto{__PIPE__L2}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L2}
  for (int b = 0; b < 32; b++) {
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L5}
    for (int f = 0; f < 128; f++) {
      output[b][f] = gamma[f] * (input[b][f] - mean[f]) / ((float )((((double )variance[f]) + 1e-5) * (((double )variance[f]) + 1e-5))) + beta[f];
    }
  }
}
