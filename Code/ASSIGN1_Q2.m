%{
                    EECE5644 FALL 2025 - ASSIGNMENT 1

                                QUESTION 2 
PART A: MINIMUM PROBABILITY OF ERROR CLASSIFICATION (0-1 LOSS, MAP
CLASSIFICATION)

%}

clear all, close all,

fprintf('\nPART A: MAP CLASSIFICATION\n');

% Creating Samples
n = 2;           % Data dimensionality
C = 4;           % Number of classes
N = 10000;       % Number of samples
p = ones(1,C)/C; % Class priors [0.25, 0.25, 0.25, 0.25]

% Mean vectors for Triangle design
mu(:,1) = [-5; -3]; % Class 1: bottom-left
mu(:,2) = [5; -3]; % Class 2: bottom-right
mu(:,3) = [0; 5]; % Class 3: top
mu(:,4) = [0; 0]; % Class 4: center (overlaps)

% Covariance matrices
Sigma(:,:,1) = [2.5  0.1; 0.1  1.0];
Sigma(:,:,2) = [2.5  0.1; 0.1  1.0];  
Sigma(:,:,3) = [2.5  0.1; 0.1  1.0];  
Sigma(:,:,4) = [4.0  0.2; 0.2  4.0];
               
fprintf('GMM Configuration:\n'); 
fprintf('  Classes: %d  |  Samples: %d  |  Dimensions: %d\n', C, N, n);
fprintf('  Class priors: [%.2f, %.2f, %.2f, %.2f]\n', p);
fprintf('  Design: Triangle (Classes 1,2,3) + Center (Class 4)\n\n');


% Generate Data - Based on sample code (generateDataFromGMM.m)
label = zeros(1,N);
u = rand(1,N); 
thresholds = [0, cumsum(p)];

for l = 1:C
    indl = find(thresholds(l) < u & u <= thresholds(l+1));
    label(indl) = l;
end

Nc = zeros(C,1);
for l = 1:C
    Nc(l) = length(find(label==l));
end

x = zeros(n,N);
for l = 1:C
    x(:,label==l) = mvnrnd(mu(:,l), Sigma(:,:,l), Nc(l))';
end

% Display data generated
figure(1), clf,
plot(x(1,label==1), x(2,label==1), 'ob'); hold on;
plot(x(1,label==2), x(2,label==2), '+r'); 
plot(x(1,label==3), x(2,label==3), '*g'); 
plot(x(1,label==4), x(2,label==4), 'xm'); 
axis equal, grid on,
xlabel('x_1'), ylabel('x_2');
title('Part A: 4-Class GMM Data (Triangle Design)');
legend('Class 1', 'Class 2', 'Class 3', 'Class 4 (Center)');

% Compute Class-Conditional Likelihoods - Based on sample code (ERMwithClabels.m)
pxgivenl = zeros(C,N);
for l = 1:C
    pxgivenl(l,:) = evalGaussian(x, mu(:,l), Sigma(:,:,l));
end

% Compute Class Posteriors - Based on sample code (ERMwithClabels.m)
px = p * pxgivenl;
classPosteriors = pxgivenl .* repmat(p', 1, N) ./ repmat(px, C, 1);

% 0-1 Loss Matrix
lambda = ones(C,C) - eye(C);

% Expected Risks and MAP Decision - Based on sample code (ERMwithClabels.m)
expectedRisks = lambda * classPosteriors;
[~, decisions_A] = min(expectedRisks, [], 1);

% Confusion Matrix - Based on sample code (ERMwithClabels.m)
ConfusionMatrix_A = zeros(C,C);
for d = 1:C
    for l = 1:C
        ind_dl = find(decisions_A==d & label==l);
        ConfusionMatrix_A(d,l) = length(ind_dl) / Nc(l);
    end
end

fprintf('Confusion Matrix P(D=d|L=l):\n');
fprintf('       L=1    L=2    L=3    L=4\n');
for d = 1:C
    fprintf('D=%d: ', d);
    fprintf(' %.4f', ConfusionMatrix_A(d,:));
    fprintf('\n');
end

% P(error)
Perror_A = sum(decisions_A ~= label) / N;
fprintf('\nP(error) = %.4f\n', Perror_A);
fprintf('Accuracy = %.2f%%\n\n', 100*(1-Perror_A));

% Visualization - Adapted from sample code (ERMwithClabels.m)
figure(2), clf,
mShapes = 'o+*x';
for d = 1:C
    for l = 1:C
        ind_dl = find(decisions_A==d & label==l);
        if d == l
            plot(x(1,ind_dl), x(2,ind_dl), ['g' mShapes(l)], 'MarkerSize', 4); 
        else
            plot(x(1,ind_dl), x(2,ind_dl), ['r' mShapes(l)], 'MarkerSize', 8);
        end
        hold on;
    end
end
axis equal, grid on,
xlabel('x_1'), ylabel('x_2');
title('Part A: MAP Classification');
legend('Correct', 'Incorrect', 'Location', 'best');

%{
PART B: ERM CLASSIFICATION WITH ASYMMETRIC LOSS MATRIX
%}

fprintf('\nPART B: ERM WITH ASYMMETRIC LOSS\n');

% Given Loss Matrix (from assignment PDF)
lambda = [0  10  10 100;
          1   0  10 100;
          1   1   0 100;
          1   1   1   0];

fprintf('Loss Matrix:\n');
disp(lambda);

% Expected Risks and ERM Decision - Based on sample code (ERMwithClabels.m)
expectedRisks = lambda * classPosteriors;
[~, decisions_B] = min(expectedRisks, [], 1);

% Confusion Matrix - Based on sample code (ERMwithClabels.m)
ConfusionMatrix_B = zeros(C,C);
for d = 1:C
    for l = 1:C
        ind_dl = find(decisions_B==d & label==l);
        ConfusionMatrix_B(d,l) = length(ind_dl) / Nc(l);
    end
end

fprintf('Confusion Matrix P(D=d|L=l):\n');
fprintf('       L=1    L=2    L=3    L=4\n');
for d = 1:C
    fprintf('D=%d: ', d);
    fprintf(' %.4f', ConfusionMatrix_B(d,:));
    fprintf('\n');
end
fprintf('\n');

% Calculate Expected Risk (Average Loss) - Based on sample code (expectedLossSampleEstimate.txt)
actualLoss = zeros(1,N);
for i = 1:N
    actualLoss(i) = lambda(decisions_B(i), label(i));
end
averageRisk = mean(actualLoss);

fprintf('Minimum Expected Risk (Sample Average): %.4f\n\n', averageRisk);

% Comparison: Part A vs Part B - adapted from discussion with Claude
fprintf(' PART A vs PART B COMPARISON\n');
fprintf('Class 4 Accuracy:\n');
fprintf('  Part A: %.4f    Part B: %.4f    Change: %+.4f\n', ...
    ConfusionMatrix_A(4,4), ConfusionMatrix_B(4,4), ...
    ConfusionMatrix_B(4,4)-ConfusionMatrix_A(4,4));

fprintf('\nOther Classes:\n');
for l = 1:3
    fprintf('  Class %d: %.4f → %.4f  (Change: %+.4f)\n', l, ...
        ConfusionMatrix_A(l,l), ConfusionMatrix_B(l,l), ...
        ConfusionMatrix_B(l,l)-ConfusionMatrix_A(l,l));
end

numChanges = sum(decisions_A ~= decisions_B);
fprintf('\nDecisions changed: %d (%.2f%% of samples)\n\n', numChanges, 100*numChanges/N);

% Visualization - Adapted from sample code (ERMwithClabels.m)
figure(3), clf,
mShapes = 'o+*x';
for d = 1:C
    for l = 1:C
        ind_dl = find(decisions_B==d & label==l);
        if d == l
            plot(x(1,ind_dl), x(2,ind_dl), ['g' mShapes(l)], 'MarkerSize', 4); 
        else
            plot(x(1,ind_dl), x(2,ind_dl), ['r' mShapes(l)], 'MarkerSize', 8);
        end
        hold on;
    end
end
axis equal, grid on,
xlabel('x_1'), ylabel('x_2');
title('Part B: ERM with Asymmetric Loss');
legend('Correct', 'Incorrect', 'Location', 'best');

% Add average risk annotation - Utilized suggestions from Copilot
text(min(x(1,:)), max(x(2,:)), sprintf('Avg Risk = %.4f', averageRisk), ...
     'VerticalAlignment', 'top', 'FontSize', 12, 'FontWeight', 'bold', ...
     'BackgroundColor', 'white');

% Side-by-side comparison - adapted from discussion with Claude
figure(4), clf,
subplot(1,2,1);
for d = 1:C
    for l = 1:C
        ind_dl = find(decisions_A==d & label==l);
        if d == l
            plot(x(1,ind_dl), x(2,ind_dl), ['g' mShapes(l)], 'MarkerSize', 4); 
        else
            plot(x(1,ind_dl), x(2,ind_dl), ['r' mShapes(l)], 'MarkerSize', 8);
        end
        hold on;
    end
end
axis equal, grid on,
xlabel('x_1'), ylabel('x_2');
title(sprintf('Part A: 0-1 Loss\nP(error) = %.4f', Perror_A));

subplot(1,2,2);
for d = 1:C
    for l = 1:C
        ind_dl = find(decisions_B==d & label==l);
        if d == l
            plot(x(1,ind_dl), x(2,ind_dl), ['g' mShapes(l)], 'MarkerSize', 4); 
        else
            plot(x(1,ind_dl), x(2,ind_dl), ['r' mShapes(l)], 'MarkerSize', 8);
        end
        hold on;
    end
end
axis equal, grid on,
xlabel('x_1'), ylabel('x_2');
title(sprintf('Part B: Asymmetric Loss\nAvg Risk = %.4f', averageRisk));

sgtitle('Comparison: MAP (Part A) vs ERM with Asymmetric Loss (Part B)');

%Code for function was taken from sample code (evalGaussian.m)
function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each column of X
[n,N] = size(x);
C = ((2*pi)^n * det(Sigma))^(-1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end