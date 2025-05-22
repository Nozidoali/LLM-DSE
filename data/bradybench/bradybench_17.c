#pragma ACCEL kernel

void top(float matrixA[64][64],float matrixB[64][64],float matrixC[64][64])
{
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (int i = 0; i < 64; i++) {
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
    for (int j = 0; j < 64; j++) {
      matrixC[i][j] = 0.0f;
    }
  }
  
#pragma ACCEL PIPELINE auto{__PIPE__L1}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
  for (int i = 0; i < 64; i++) {
    
#pragma ACCEL PIPELINE auto{__PIPE__L3}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L3}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L3}
    for (int j = 0; j < 64; j++) {
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L4}
      for (int k = 0; k < 64; k++) {
        matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
      }
    }
  }
}
