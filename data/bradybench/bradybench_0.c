#pragma ACCEL kernel

void top(double positions[100][3],double velocities[100][3],double forces[100][3])
{
  double invDist;
  double distSqr;
  double dz;
  double dy;
  double dx;
  double mass;
  double dt;
  dt = 0.01;
  mass = 1.0;
  int i;
  int j;
  int k;
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (i = 0; i < 100; ++i) {
    for (j = 0; j < 3; ++j) {
      forces[i][j] = 0.0;
    }
  }
  
#pragma ACCEL PIPELINE auto{__PIPE__L1}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
  for (k = 0; k < 10; ++k) {
    
#pragma ACCEL PIPELINE auto{__PIPE__L3}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L3}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L3}
    for (i = 0; i < 100; ++i) {
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L5}
      for (j = 0; j < 100; ++j) {
        dx = positions[j][0] - positions[i][0];
        dy = positions[j][1] - positions[i][1];
        dz = positions[j][2] - positions[i][2];
        distSqr = dx * dx + dy * dy + dz * dz + 0.1;
        invDist = 1.0 / distSqr;
        forces[i][0] += invDist * dx;
        forces[i][1] += invDist * dy;
        forces[i][2] += invDist * dz;
      }
    }
    
#pragma ACCEL PIPELINE auto{__PIPE__L4}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L4}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L4}
    for (i = 0; i < 100; ++i) {
      for (j = 0; j < 3; ++j) {
        velocities[i][j] += dt * forces[i][j] / mass;
        positions[i][j] += dt * velocities[i][j];
      }
    }
  }
}
