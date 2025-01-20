# pragma ACCEL kernel

void loop(int tmp[6])
{
#pragma ACCEL PARALLEL auto{__PARA__L0}
L0:    for (int i = 0; i < 128; i++)
    {
        tmp[i] += 1;
    }
}