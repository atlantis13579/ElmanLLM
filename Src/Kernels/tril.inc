// tril scalar implementation.
int columns=P(2+P(1)-1);int rows=P(2+P(1)-2);int row=(i/columns)%rows;int column=i%columns;
if(column-row<=P(64)){COPY_A(i,i);}else{W(i,0.0f);}
