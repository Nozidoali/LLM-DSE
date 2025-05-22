#pragma ACCEL kernel

void top(double matrix[100][100],double vector[100],double result[100])
{
  int i;
  int j;
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (i = 0; i < 100; ++i) {
    result[i] = 0.0;
  }
  
#pragma ACCEL PIPELINE auto{__PIPE__L1}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
  for (i = 0; i < 100; ++i) {
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
    for (j = 0; j < 100; ++j) {
      result[i] += matrix[i][j] * vector[j];
    }
  }
}
