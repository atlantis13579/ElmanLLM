// slice_backward scalar implementation.
int flat=i;int source=0;int valid=1;
for(int axis=P(1)-1;axis>=0;--axis){
    int coordinate=flat%P(2+axis);flat/=P(2+axis);
    if(axis==P(64)){coordinate-=P(65);if(coordinate<0||coordinate%P(66)!=0)valid=0;coordinate/=P(66);if(coordinate>=P(11+axis))valid=0;}
    source+=coordinate*P(19+axis);
}
W(i,valid!=0?A(source):0.0f);
