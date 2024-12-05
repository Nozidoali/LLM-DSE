#pragma ACCEL kernel

void top(float input[3][32][32],float kernel[8][3][3][3],float output[8][30][30])
{
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  for (int f = 0; f < 8; f++) {
    
#pragma ACCEL PIPELINE auto{__PIPE__L2}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L2}
    for (int i = 0; i < 32 - 3 + 1; i++) {
      for (int j = 0; j < 32 - 3 + 1; j++) {
        output[f][i][j] = 0.0f;
      }
    }
  }
  
#pragma ACCEL PIPELINE auto{__PIPE__L1}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
  for (int f = 0; f < 8; f++) {
    
#pragma ACCEL PIPELINE auto{__PIPE__L3}
    for (int c = 0; c < 3; c++) {
      
#pragma ACCEL PIPELINE auto{__PIPE__L5}
      
#pragma ACCEL TILE FACTOR=auto{__TILE__L5}
      for (int i = 0; i < 32 - 3 + 1; i++) {
        
#pragma ACCEL PIPELINE auto{__PIPE__L6}
        
#pragma ACCEL TILE FACTOR=auto{__TILE__L6}
        for (int j = 0; j < 32 - 3 + 1; j++) {
          
#pragma ACCEL PIPELINE auto{__PIPE__L7}
          for (int ki = 0; ki < 3; ki++) {
            for (int kj = 0; kj < 3; kj++) {
              output[f][i][j] += input[c][i + ki][j + kj] * kernel[f][c][ki][kj];
            }
          }
        }
      }
    }
  }
}
