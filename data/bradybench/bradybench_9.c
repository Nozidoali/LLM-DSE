#pragma ACCEL kernel

void top(double hamiltonian[256][256],double density[256][256],double energy[256])
{
  double row_sum;
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (int i = 0; i < 256; i++) {
    energy[i] = 0.0;
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
    for (int j = 0; j < 256; j++) {
      density[i][j] = 0.0;
    }
  }
  
#pragma ACCEL PIPELINE auto{__PIPE__L1}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
  for (int iter = 0; iter < 10; iter++) {
    
#pragma ACCEL PIPELINE auto{__PIPE__L3}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L3}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L3}
    for (int i = 0; i < 256; i++) {
      
#pragma ACCEL PIPELINE auto{__PIPE__L5}
      
#pragma ACCEL TILE FACTOR=auto{__TILE__L5}
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L5}
      for (int j = 0; j < 256; j++) {
        density[i][j] = 0.0;
        
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L7}
        for (int k = 0; k < 256; k++) {
          density[i][j] += hamiltonian[i][k] * hamiltonian[k][j];
        }
      }
    }
    
#pragma ACCEL PIPELINE auto{__PIPE__L4}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L4}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L4}
    for (int i = 0; i < 256; i++) {
      row_sum = 0.0;
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L6}
      for (int j = 0; j < 256; j++) {
        row_sum += density[i][j] * hamiltonian[i][j];
      }
      energy[i] = row_sum;
    }
  }
}
