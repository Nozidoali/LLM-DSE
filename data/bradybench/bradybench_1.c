#pragma ACCEL kernel

void top(double grid[50][50],double new_grid[50][50])
{
  double alpha;
  int i;
  int j;
  int k;
  alpha = 0.01;
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (i = 0; i < 50; ++i) {
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
    for (j = 0; j < 50; ++j) {
      new_grid[i][j] = 0.0;
    }
  }
  
#pragma ACCEL PIPELINE auto{__PIPE__L1}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
  for (k = 0; k < 20; ++k) {
    
#pragma ACCEL PIPELINE auto{__PIPE__L3}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L3}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L3}
    for (i = 1; i < 50 - 1; ++i) {
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L5}
      for (j = 1; j < 50 - 1; ++j) {
        new_grid[i][j] = grid[i][j] + alpha * (grid[i + 1][j] - ((double )2) * grid[i][j] + grid[i - 1][j] + (grid[i][j + 1] - ((double )2) * grid[i][j] + grid[i][j - 1]));
      }
    }
    
#pragma ACCEL PIPELINE auto{__PIPE__L4}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L4}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L4}
    for (i = 0; i < 50; ++i) {
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L6}
      for (j = 0; j < 50; ++j) {
        grid[i][j] = new_grid[i][j];
      }
    }
  }
}
