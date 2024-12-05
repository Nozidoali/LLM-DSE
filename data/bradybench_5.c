#pragma ACCEL kernel

void top(float radiative_flux_up[80][80][10],float radiative_flux_down[80][80][10],float absorption[80][80][10],float scattering[80][80][10],float reflectivity[80][80][10])
{
  float balance;
  float new_flux_up[80][80][10];
  float new_flux_down[80][80][10];
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  for (int t = 0; t < 5; t++) {
    
#pragma ACCEL PIPELINE auto{__PIPE__L1}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
    for (int z = 1; z < 10 - 1; z++) {
      
#pragma ACCEL PIPELINE auto{__PIPE__L4}
      
#pragma ACCEL TILE FACTOR=auto{__TILE__L4}
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L4}
      for (int i = 0; i < 80; i++) {
        
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L7}
        for (int j = 0; j < 80; j++) {
          new_flux_up[i][j][z] = radiative_flux_up[i][j][z] * (((float )1) - absorption[i][j][z]) + radiative_flux_down[i][j][z] * scattering[i][j][z] + radiative_flux_up[i][j][z - 1] * reflectivity[i][j][z];
          new_flux_down[i][j][z] = radiative_flux_down[i][j][z] * (((float )1) - absorption[i][j][z]) + radiative_flux_up[i][j][z] * scattering[i][j][z] + radiative_flux_down[i][j][z + 1] * reflectivity[i][j][z];
        }
      }
    }
    
#pragma ACCEL PIPELINE auto{__PIPE__L2}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L2}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
    for (int i = 0; i < 80; i++) {
      
#pragma ACCEL PIPELINE auto{__PIPE__L5}
      
#pragma ACCEL TILE FACTOR=auto{__TILE__L5}
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L5}
      for (int j = 0; j < 80; j++) {
        new_flux_up[i][j][0] = radiative_flux_up[i][j][0];
        new_flux_down[i][j][10 - 1] = radiative_flux_down[i][j][10 - 1];
        for (int z = 0; z < 10; z++) {
          radiative_flux_up[i][j][z] = new_flux_up[i][j][z];
          radiative_flux_down[i][j][z] = new_flux_down[i][j][z];
        }
      }
    }
    
#pragma ACCEL PIPELINE auto{__PIPE__L3}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L3}
    for (int z = 0; z < 10; z++) {
      
#pragma ACCEL PIPELINE auto{__PIPE__L6}
      
#pragma ACCEL TILE FACTOR=auto{__TILE__L6}
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L6}
      for (int i = 0; i < 80; i++) {
        
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L9}
        for (int j = 0; j < 80; j++) {
          balance = radiative_flux_up[i][j][z] - radiative_flux_down[i][j][z];
          absorption[i][j][z] = (balance > ((float )0)?balance * 0.1f : ((float )0));
        }
      }
    }
  }
}
