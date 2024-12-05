#pragma ACCEL kernel

void top(float input[16][20][64],float hidden_state[16][128],float weights_input[64][128],float weights_hidden[128][128],float bias[128],float output[16][20][128])
{
  float sum;
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  for (int b = 0; b < 16; b++) {
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
    for (int h = 0; h < 128; h++) {
      hidden_state[b][h] = 0.0f;
    }
  }
  
#pragma ACCEL PIPELINE auto{__PIPE__L1}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
  for (int t = 0; t < 20; t++) {
    
#pragma ACCEL PIPELINE auto{__PIPE__L3}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L3}
    for (int b = 0; b < 16; b++) {
      
#pragma ACCEL PIPELINE auto{__PIPE__L4}
      
#pragma ACCEL TILE FACTOR=auto{__TILE__L4}
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L4}
      for (int h = 0; h < 128; h++) {
        sum = bias[h];
        
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L6}
        for (int f = 0; f < 64; f++) {
          sum += input[b][t][f] * weights_input[f][h];
        }
        
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L7}
        for (int hh = 0; hh < 128; hh++) {
          sum += hidden_state[b][hh] * weights_hidden[hh][h];
        }
        hidden_state[b][h] = (sum > ((float )0)?sum : ((float )0));
      }
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L5}
      for (int h = 0; h < 128; h++) {
        output[b][t][h] = hidden_state[b][h];
      }
    }
  }
}
