extern int max();
extern int max();
extern int max();
extern int max();
#pragma ACCEL kernel

void CnnKernel(const float input[256][228][228],const float weight[256][256][5][5],const float bias[256],float output[256][112][112])
{
// Allocate memory on heap to avoid stack overflow.
  static float C[256][224][224];
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (int i = 0; i < 256; ++i) {
    
#pragma ACCEL PIPELINE auto{__PIPE__L4}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L4}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L4}
    for (int h = 0; h < 224; ++h) {
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L8}
      for (int w = 0; w < 224; ++w) {
        C[i][h][w] = bias[i];
      }
    }
  }
// Convolution
  
#pragma ACCEL PIPELINE auto{__PIPE__L1}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
  for (int i = 0; i < 256; ++i) {
    
#pragma ACCEL PIPELINE auto{__PIPE__L5}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L5}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L5}
    for (int j = 0; j < 256; ++j) {
      
#pragma ACCEL PIPELINE auto{__PIPE__L9}
      
#pragma ACCEL TILE FACTOR=auto{__TILE__L9}
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L9}
      for (int h = 0; h < 224; ++h) {
        
#pragma ACCEL PIPELINE auto{__PIPE__L12}
        
#pragma ACCEL TILE FACTOR=auto{__TILE__L12}
        
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L12}
        for (int w = 0; w < 224; ++w) {
          
#pragma ACCEL PIPELINE auto{__PIPE__L13}
          
#pragma ACCEL TILE FACTOR=auto{__TILE__L13}
          for (int p = 0; p < 5; ++p) {
            for (int q = 0; q < 5; ++q) {
              C[i][h][w] += weight[i][j][p][q] * input[j][h + p][w + q];
            }
          }
        }
      }
    }
  }
// ReLU
  
#pragma ACCEL PIPELINE auto{__PIPE__L2}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L2}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
  for (int i = 0; i < 256; ++i) {
    
#pragma ACCEL PIPELINE auto{__PIPE__L6}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L6}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L6}
    for (int h = 0; h < 224; ++h) {
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L10}
      for (int w = 0; w < 224; ++w) {
        C[i][h][w] = ((float )(max(0,(double )C[i][h][w])));
      }
    }
  }
// Max pooling
  
#pragma ACCEL PIPELINE auto{__PIPE__L3}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L3}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L3}
  for (int i = 0; i < 256; ++i) {
    
#pragma ACCEL PIPELINE auto{__PIPE__L7}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L7}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L7}
    for (int h = 0; h < 112; ++h) {
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L11}
      for (int w = 0; w < 112; ++w) {
        output[i][h][w] = ((float )(max((max((double )C[i][h * 2][w * 2],(double )C[i][h * 2 + 1][w * 2])),(max((double )C[i][h * 2][w * 2 + 1],(double )C[i][h * 2 + 1][w * 2 + 1])))));
      }
    }
  }
}
