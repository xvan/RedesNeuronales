pkg load statistics
clear all
close all

ImageNames45x60={
  "images/paloma.bmp";
  "images/quijote.bmp" ;
  "images/torero.bmp" ;
};

ImageNames50x50={  
  "images/panda.bmp" ;
  "images/perro.bmp" ; 
  "images/v.bmp" ;
};

%Funcion Signo sin cero
function out=defsgn(val)
  out=(val >= 0)*2-1;
endfunction

%Hopfield Iteration
function out=hopfield(seed,w)
  out=seed;
  iteration=1;
  changed=false;
    
  imshow(reshape(out,45,60));
  do    
    if mod(iteration,1)== 0
      imshow(reshape(out,45,60));
      drawnow;
      pause(0.3)
    endif
    
    %disp(iteration++);
    changed=false;    
    for i=randperm(numel(out))
      hi=defsgn(w(i,:)*out');      
      changed |= (hi ~= out(i));
      out(i)=hi;
    endfor      
  until ~changed
endfunction


function Images=LoadImages(ImageFileNames)
  %Carga de las Imagenes  
  Images=[];
  for idx = 1:numel(ImageFileNames)
    Images=[Images; reshape(imread(ImageFileNames{idx})*2-1,1,[])];
  endfor
endfunction

function W=CalculateWeights(Images)
  W=0;
  for idx = 1:size(Images)(1)
    W+=Images(idx,:)'*Images(idx,:)-eye(size(Images)(2));
  endfor
endfunction

function Learned=AutoTest(SourceImages, TargetImages, W)
  %chequeo si aprendio evaluando las mismas imagenes.
  Learned=[];
  for idx = 1:size(SourceImages)(1)
    Learned=[Learned, all(hopfield(SourceImages(idx,:),W) == TargetImages(idx,:))];
  endfor
endfunction


function ImagesWithNoise=AddNoise(Images,level)
  ImagesWithNoise = Images .* ( rand(size(Images)) > level) * 2 - 1;
endfunction

function detected=TestNoise(Images,W)
  detected=[];
  for lvl=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    ImagesWithNoise = AddNoise(Images,lvl);
    detected=[ detected; AutoTest(ImagesWithNoise,Images,W)];
  endfor
endfunction

function mask=CreateMask(N,M,lvl)
  mask=zeros(N,M);
  mask(1:floor(N*lvl),1:floor(M*lvl))=1;
endfunction


function MaskedImages=MaskImages(Images,N,M,lvl)
  mask=CreateMask(N,M,lvl);  
  MaskedImages=(Images + repmat(reshape(mask,1,[]),size(Images)(1),1) >= 0 )*2-1 ;
endfunction

function detected=TestMasked(Images,W)
  detected=[];
  for lvl=[0.2:0.2:0.8]
    MaskedImages = MaskImages(Images,45,60,lvl);
    detected=[ detected; AutoTest(MaskedImages,Images,W)];
  endfor
endfunction


function SpuriousImages=GenerateOddSpurious(Images)
  ExtendedImages=[Images ; -Images];
  v=1:size(ExtendedImages)(1);
  c=combnk(v,3);
  SpuriousImages=zeros(size(c)(1),size(ExtendedImages)(2));
  for combIdx = 1:length(c)    
    for imgIdx = 1:3
      SpuriousImages(combIdx,:)+=ExtendedImages(c(combIdx,imgIdx),:);
    endfor    
  endfor
  SpuriousImages=(SpuriousImages>0)*2-1;
endfunction

function detected=TestSpurious(Images,W)
  SpuriousImages=GenerateOddSpurious(Images);
  detected=AutoTest(SpuriousImages,SpuriousImages,W);
endfunction


ImageFileNames=ImageNames45x60;
Images=LoadImages(ImageFileNames);
W=CalculateWeights(Images);  
%Learned=AutoTest(Images,Images,W)
%DetectedNoise=TestNoise(Images,W)
%DetectedMasked=TestMasked(Images,W)
DetectedNegativeSpurious=TestSpurious(Images,W)

%Learned45x60=testHopfield(ImageNames45x60)
%Learned50x50=testHopfield(ImageNames50x50)

