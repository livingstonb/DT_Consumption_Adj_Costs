
smallyPtrans = discmodel1.yPtrans - diag(diag(discmodel1.yPtrans));
smallyPtrans = smallyPtrans(4:7,4:7);
for i = 1:4
    smallyPtrans(i,i) = - sum(smallyPtrans(i,:));
end

discmodel1.yPtrans = smallyPtrans;
discmodel1.yPgrid = discmodelyPgrid(4:7);