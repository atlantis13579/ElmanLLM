// transpose scalar implementation.
int flat=i;int source=0;
for(int axis=P(1)-1;axis>=0;--axis){int coordinate=flat%P(2+axis);flat/=P(2+axis);int source_axis=axis==P(64)?P(65):(axis==P(65)?P(64):axis);source+=coordinate*P(19+source_axis);}
COPY_A(i,source);
