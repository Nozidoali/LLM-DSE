#pragma ACCEL kernel

void top(float wind_u[80][80][10],float wind_v[80][80][10],float wind_w[80][80][10],float temperature[80][80][10],float pressure[80][80][10])
{
  float new_wind_u[80][80][10];
  float new_wind_v[80][80][10];
  float new_wind_w[80][80][10];
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  for (int t = 0; t < 5; t++) {
    
#pragma ACCEL PIPELINE auto{__PIPE__L1}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
    for (int i = 1; i < 80 - 1; i++) {
      
#pragma ACCEL PIPELINE auto{__PIPE__L6}
      
#pragma ACCEL TILE FACTOR=auto{__TILE__L6}
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L6}
      for (int j = 1; j < 80 - 1; j++) {
        for (int z = 0; z < 10; z++) {
          new_wind_u[i][j][z] = wind_u[i][j][z] + 0.1f * (wind_u[i + 1][j][z] - ((float )2) * wind_u[i][j][z] + wind_u[i - 1][j][z]) + 0.1f * (wind_u[i][j + 1][z] - ((float )2) * wind_u[i][j][z] + wind_u[i][j - 1][z]);
          new_wind_v[i][j][z] = wind_v[i][j][z] + 0.1f * (wind_v[i + 1][j][z] - ((float )2) * wind_v[i][j][z] + wind_v[i - 1][j][z]) + 0.1f * (wind_v[i][j + 1][z] - ((float )2) * wind_v[i][j][z] + wind_v[i][j - 1][z]);
        }
      }
    }
    
#pragma ACCEL PIPELINE auto{__PIPE__L2}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L2}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
    for (int i = 1; i < 80 - 1; i++) {
      
#pragma ACCEL PIPELINE auto{__PIPE__L7}
      
#pragma ACCEL TILE FACTOR=auto{__TILE__L7}
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L7}
      for (int j = 1; j < 80 - 1; j++) {
        for (int z = 1; z < 10 - 1; z++) {
          new_wind_w[i][j][z] = wind_w[i][j][z] + 0.1f * (pressure[i][j][z - 1] - pressure[i][j][z + 1]);
        }
      }
    }
    
#pragma ACCEL PIPELINE auto{__PIPE__L3}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L3}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L3}
    for (int i = 1; i < 80 - 1; i++) {
      
#pragma ACCEL PIPELINE auto{__PIPE__L8}
      
#pragma ACCEL TILE FACTOR=auto{__TILE__L8}
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L8}
      for (int j = 1; j < 80 - 1; j++) {
        for (int z = 0; z < 10; z++) {
          temperature[i][j][z] += 0.1f * (wind_u[i][j][z] * (temperature[i + 1][j][z] - temperature[i - 1][j][z]) + wind_v[i][j][z] * (temperature[i][j + 1][z] - temperature[i][j - 1][z]) + wind_w[i][j][z] * (temperature[i][j][z + 1] - temperature[i][j][z - 1]));
        }
      }
    }
    
#pragma ACCEL PIPELINE auto{__PIPE__L4}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L4}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L4}
    for (int i = 0; i < 80; i++) {
      
#pragma ACCEL PIPELINE auto{__PIPE__L9}
      
#pragma ACCEL TILE FACTOR=auto{__TILE__L9}
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L9}
      for (int j = 0; j < 80; j++) {
        for (int z = 0; z < 10; z++) {
          wind_u[i][j][z] = new_wind_u[i][j][z];
          wind_v[i][j][z] = new_wind_v[i][j][z];
          wind_w[i][j][z] = new_wind_w[i][j][z];
        }
      }
    }
    
#pragma ACCEL PIPELINE auto{__PIPE__L5}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L5}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L5}
    for (int i = 1; i < 80 - 1; i++) {
      
#pragma ACCEL PIPELINE auto{__PIPE__L10}
      
#pragma ACCEL TILE FACTOR=auto{__TILE__L10}
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L10}
      for (int j = 1; j < 80 - 1; j++) {
        for (int z = 0; z < 10; z++) {
          pressure[i][j][z] -= 0.1f * (temperature[i][j][z] - temperature[i][j][z + 1]);
        }
      }
    }
  }
}
