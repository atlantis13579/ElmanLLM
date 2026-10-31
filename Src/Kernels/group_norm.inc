// group_norm scalar implementation.
int channels=P(12);int spatial=P(20);int per_group=channels/P(64);int length=per_group*spatial;int base=(i/length)*length;
float average=0.0f;for(int j=0;j<length;++j)average+=A(base+j);average/=float(length);
float variance=0.0f;for(int j=0;j<length;++j){float difference=A(base+j)-average;variance+=difference*difference;}variance/=float(length);
float value=(A(i)-average)/EL_SQRT(variance+F(65));int channel=(i/spatial)%channels;
if(P(27)>=0)value*=B(channel);if(P(44)>=0)value+=C(channel);W(i,value);
