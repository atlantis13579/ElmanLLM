// repeat_grad scalar implementation.
float value=0.0f;int count=EL_COUNT(10,11);int delta=P(10)-P(1);
for(int j=0;j<count;++j){int flat=j;int target=0;int stride=1;
    for(int axis=P(10)-1;axis>=0;--axis){int coordinate=flat%P(11+axis);flat/=P(11+axis);int target_axis=axis-delta;if(target_axis>=0){target+=(coordinate%P(2+target_axis))*stride;stride*=P(2+target_axis);}}
    if(target==i)value+=A(j);
}
W(i,value);
