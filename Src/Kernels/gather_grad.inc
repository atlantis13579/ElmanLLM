// gather_grad scalar implementation.
float value=0.0f;
int count=EL_COUNT(10,11);
for(int j=0;j<count;++j){int flat=j;int target=0;int stride=1;int selected=IB(j);
    for(int axis=P(1)-1;axis>=0;--axis){int coordinate=flat%P(11+axis);flat/=P(11+axis);if(axis==P(64))coordinate=selected;target+=coordinate*stride;stride*=P(2+axis);}
    if(target==i&&selected>=0&&selected<P(2+P(64)))value+=A(j);
}
W(i,value);
