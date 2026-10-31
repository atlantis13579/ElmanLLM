// layer_norm scalar implementation.
int length=1;for(int axis=P(10)-P(64);axis<P(10);++axis)length*=P(11+axis);
int base=(i/length)*length;float average=0.0f;
for(int j=0;j<length;++j)average+=A(base+j);average/=float(length);
float variance=0.0f;for(int j=0;j<length;++j){float difference=A(base+j)-average;variance+=difference*difference;}variance/=float(length);
float value=(A(i)-average)/EL_SQRT(variance+F(65));int coordinate=i%length;
if(P(27)>=0)value*=B(coordinate);if(P(44)>=0)value+=C(coordinate);
W(i,value);
