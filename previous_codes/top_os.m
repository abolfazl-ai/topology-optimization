%%%% A 199 LINES CODE FOR MULTI-MATERIALS TOPOLOGY OPTIMIZATION %%%
%%%%%%%%% BY WENJIE ZUO AND KAZUHIRO SAITOU, DEC. 2015 %%%%%%%%
function top_os(nelx,nely,X0,volfrac,costfrac,penal,rmin,D,E,P,MColor,MName,MinMove)
% INITIALIZE
x(1:nely,1:nelx) = X0; 
loop = 0; 
change = 1.;
% START ITERATION
while change >1.01*MinMove  
  loop = loop + 1;
  xold = x;
% FE-ANALYSIS
  [E_,dE_]=OrderedSIMPInterpolation(nelx,nely,x,penal,D,E);
  [P_,dP_]=OrderedSIMPInterpolation(nelx,nely,x,1/penal,D,P);
  [U]=FE(nelx,nely,E_);         
% OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
  [KE] = lk;
  c = 0.;
  dc=zeros(nely,nelx);
  for ely = 1:nely
    for elx = 1:nelx
      n1 = (nely+1)*(elx-1)+ely; 
      n2 = (nely+1)* elx   +ely;
      Ue = U([2*n1-1;2*n1; 2*n2-1;2*n2; 2*n2+1;2*n2+2; 2*n1+1;2*n1+2],1);
      c = c + E_(ely,elx)*Ue'*KE*Ue;
      dc(ely,elx) = -dE_(ely,elx)*Ue'*KE*Ue;
    end
  end
% FILTERING OF SENSITIVITIES
  [dc]=check(nelx,nely,rmin,x,dc);    
% DESIGN UPDATE BY THE OPTIMALITY CRITERIA METHOD
  [x]=OC(nelx,nely,x,volfrac,costfrac,dc,E_,dE_,P_,dP_,loop,MinMove); 
% PRINT RESULTS
  change = max(max(abs(x-xold)));
  disp([' It.: ' sprintf('%i',loop) '  Obj.: ' sprintf('%.4f',c) ...
       '  Mass Fraction: ' sprintf('%.3f',sum(sum(x))/(nelx*nely)) ...
       '  Cost Fraction: ' sprintf('%.3f', sum(sum(x.*P_))/(nelx*nely)) ...
        '  Ch.: ' sprintf('%.3f',change )])
% RENDER MULTI-MATERIAL TOPOLOGY  
  Render(nelx,nely,x,D,MColor,MName)
  axis equal; axis tight; axis off;pause(1e-6);
  saveas(gcf,sprintf('%i',loop),'tif');
  Objective(loop)=c;
  MassFraction(loop)=sum(sum(x))/(nelx*nely);
  CostFraction(loop)=sum(sum(x.*P_))/(nelx*nely);
end 
  save Objective.txt Objective -ascii
  save MassFraction.txt MassFraction -ascii
  save CostFraction.txt CostFraction -ascii
end
%%%%%%%% OPTIMALITY CRITERIA UPDATE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [xnew]=OC(nelx,nely,x,volfrac,costfrac,dc,E_,dE_,P_,dP_,loop,MinMove)
dc=-1*dc;
lV1 = 0;
lV2=2*max(max(dc)); 
Temp=P_+x.*dP_;
Temp=dc./Temp;
lP1 = 0;
lP2=2*max(max(Temp)); 
move = max(0.15*0.96^loop,MinMove);
while(((lV2-lV1)/(lV1+lV2)> 1e-6) || ((lP2-lP1)/(lP1+lP2)> 1e-6))
  lmidV = 0.5*(lV2+lV1);
  lmidP = 0.5*(lP2+lP1);
  Temp=lmidV+lmidP.*P_+lmidP*x.*dP_;
  Coef=dc./Temp;
  Coef=abs(Coef);
  xnew = max(10^-5,max(x-move,min(1.,min(x+move,x.*sqrt(Coef)))));
  if sum(sum(xnew)) - volfrac*nelx*nely > 0;
    lV1 = lmidV;
  else
    lV2 = lmidV;
  end
  CurrentCostFrac=sum(sum(xnew.*P_))/(nelx*nely);
  if CurrentCostFrac - costfrac > 0;
    lP1 = lmidP;
  else
    lP2 = lmidP;
  end
end
end
%%%%%%%% MESH-INDEPENDENCY FILTER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [dcn]=check(nelx,nely,rmin,x,dc)
dcn=zeros(nely,nelx);
for i = 1:nelx
  for j = 1:nely
    sum=0.0; 
    for k = max(i-floor(rmin),1):min(i+floor(rmin),nelx)
      for l = max(j-floor(rmin),1):min(j+floor(rmin),nely)
        fac = rmin-sqrt((i-k)^2+(j-l)^2);
        sum = sum+max(0,fac);
        dcn(j,i) = dcn(j,i) + max(0,fac)*x(l,k)*dc(l,k);
      end
    end
    dcn(j,i) = dcn(j,i)/(x(j,i)*sum);
  end
end
end
%%%%%%%% FE-ANALYSIS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [U]=FE(nelx,nely,E_Intopolation)
[KE] = lk; 
K = sparse(2*(nelx+1)*(nely+1), 2*(nelx+1)*(nely+1));
F = sparse(2*(nely+1)*(nelx+1),1); 
U = zeros(2*(nely+1)*(nelx+1),1);
for elx = 1:nelx
  for ely = 1:nely
    n1 = (nely+1)*(elx-1)+ely; 
    n2 = (nely+1)* elx   +ely;
    edof = [2*n1-1; 2*n1; 2*n2-1; 2*n2; 2*n2+1; 2*n2+2; 2*n1+1; 2*n1+2];
    K(edof,edof) = K(edof,edof) + E_Intopolation(ely,elx)*KE;
  end
end
% DEFINE LOADS AND SUPPORTS (Bridge)
F(2*(nely+1)*(nelx/4+1),1) = -1;
F(2*(nely+1)*(2*nelx/4+1),1) = -2;
F(2*(nely+1)*(3*nelx/4+1),1) = -1;
fixeddofs = union([2*(nely+1)-1;2*(nely+1)],[2*(nelx+1)*(nely+1)]);
alldofs = [1:2*(nely+1)*(nelx+1)];
freedofs = setdiff(alldofs,fixeddofs);
% SOLVING
U(freedofs,:) = K(freedofs,freedofs) \ F(freedofs,:);      
U(fixeddofs,:)= 0;
end
%%%%%%%% ORDERED SIMP INTERPOLATION AND ITS DERIVATIVE %%%%%%%%%%%%%%%%%%%%
function [y,dy]=OrderedSIMPInterpolation(nelx,nely,x,penal,X,Y)
y=zeros(nely,nelx);
dy=zeros(nely,nelx);
for i = 1:nelx
  for j = 1:nely
    for k=1:length(X)-1
        if (X(k)<x(j,i)) && (X(k+1)>=x(j,i))
            A=(Y(k)-Y(k+1))/(X(k)^(1*penal)-X(k+1)^(1*penal));
            B=Y(k)-A*(X(k)^(1*penal));
            y(j,i)=A*(x(j,i)^(1*penal))+B;
            dy(j,i)=A*penal*(x(j,i)^((1*penal)-1));
            break;
        end
    end
  end
end
end
%%%%%%%% ELEMENT STIFFNESS MATRIX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [KE]=lk
E = 1.;nu = 0.3;
A11 = [12  3 -6 -3;  3 12  3  0; -6  3 12 -3; -3  0 -3 12];
A12 = [-6 -3  0  3; -3 -6 -3 -6;  0 -3 -6  3;  3 -6  3 -6];
B11 = [-4  3 -2  9;  3 -4 -9  4; -2 -9 -4 -3;  9  4 -3 -4];
B12 = [ 2 -3  4 -9; -3  2  9 -2;  4  9  2  3; -9 -2  3  2];
KE = E/(1-nu^2)/24*([A11 A12;A12' A11]+nu*[B11 B12;B12' B11]);
end
%%%%%%%% RENDER MULTI-MATERIAL TOPOLOGY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Render(nelx,nely,x,D,TColor,Name)
% MEDIAN MATERIAL DENSITY
MedianD=D;
for i=1 : length(MedianD)-1
    MedianD(i)=0.5*(D(i)+D(i+1));
end
% DEFINITION OF IMAGE SIZE AND DATA WRITING
ES=8;LS=1; % ELEMENT SIZE ES;LINE SIZE LS
m1=nely*ES+(nely+1)*LS;n1=nelx*ES+(nelx+1)*LS;
Image=zeros(m1,n1,3);
for i=1:nely
    for j=1:nelx
        for k=1:length(MedianD)
            if x(i,j)<=MedianD(k)
                RGB=TextToRGB(TColor(k));
                break;
            end
        end
        for k=i*LS+1+(i-1)*ES:i*LS+1+i*ES
            for l=j*LS+1+(j-1)*ES:j*LS+1+j*ES
               Image(k,l,1)=RGB(1);
               Image(k,l,2)=RGB(2);
               Image(k,l,3)=RGB(3);
            end
        end
    end
end
%CREATE LEGENDS
h=50;rs=18;dis=165;len=180;% LEGEND SIZE CONTROL
Title=201*ones(h,n1,3);% GRAY LEGEND BACKGROUND
MergeImage=[Title;Image]./255;
imshow(MergeImage);
for i=1:length(MedianD)
    text(len+rs+6+(i-1)*dis,0.5*h,Name(i),'color',TColor(i),'Fontsize',rs*0.8);
    rectangle('Position',[len+(i-1)*dis,(h-rs)/2,rs,rs],'FaceColor',TColor(i));
end
end
%%%%%%%% TRANSFER TEXT INTO RGB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [RGB]=TextToRGB(TColor)
if TColor=='w' % WHITE
    RGB(1)=255;   RGB(2)=255;   RGB(3)=255;
elseif TColor=='b' % BLUE
    RGB(1)=0;   RGB(2)=0;   RGB(3)=255;
elseif TColor=='k' % BLACK
    RGB(1)=0;   RGB(2)=0;   RGB(3)=0;
elseif TColor=='r' % RED
    RGB(1)=255;   RGB(2)=0;   RGB(3)=0;
end
end

