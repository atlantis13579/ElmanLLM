// repeat scalar implementation.
int flat=i;int source=0;int delta=P(1)-P(10);
for(int axis=P(1)-1;axis>=0;--axis){int coordinate=flat%P(2+axis);flat/=P(2+axis);int source_axis=axis-delta;if(source_axis>=0)source+=(coordinate%P(11+source_axis))*P(19+source_axis);}
COPY_A(i,source);
