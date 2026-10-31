// permute scalar implementation.
int flat=i;int source=0;
for(int axis=P(1)-1;axis>=0;--axis){int coordinate=flat%P(2+axis);flat/=P(2+axis);source+=coordinate*P(19+P(64+axis));}
COPY_A(i,source);
