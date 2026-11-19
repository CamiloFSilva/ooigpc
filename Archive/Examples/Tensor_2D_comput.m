clear all

num=4;

syms x
syms

psi0 = 1;
psi1 = x;
psi2 = x^2-1;


PSIA = [psi0, psi1, psi0, psi1];
PSIB = [psi0, psi0, psi1, psi1];


PSI = PSIA.*PSIB;

prob = 1/(sqrt(2*pi))*exp(-x^2/2);

%%

for k=1:num

    funcA = PSIA(k)*PSIA(k); funcB = PSIB(k)*PSIB(k);

    normA = eval(int(funcA*prob,-inf,inf));
    normB = eval(int(funcB*prob,-inf,inf)) ;

    norm(k) = normA*normB ;

end


% 
% for k=1:num
%     for j=1:num
%         for i=1:num
% 
%             funcA = PSIA(i)*PSIA(j)*PSIA(k); funcB = PSIB(i)*PSIB(j)*PSIB(k);
% 
%             ctA = eval(int(funcA*prob,-inf,inf));
%             ctB = eval(int(funcB*prob,-inf,inf));
% 
%             Ctensor(i,j,k) = ctA*ctB;
% 
%         end
%     end
% 
% end


for i=1:num
    for j=1:num
        for k=1:num
            for l=1:3

                if l==1

                    funcAC = PSIA(i)*PSIA(j)*PSIA(k); funcBC = PSIB(i)*PSIB(j)*PSIB(k);

                    ctA = eval(int(funcAC*prob,-inf,inf));
                    ctB = eval(int(funcBC*prob,-inf,inf));

                    Ctensor{k}(i,j) = sparse(ctA*ctB);

                end

                funcA = PSIA(i)*PSIA(j)*PSIA(k)*PSIA(l);
                funcB = PSIB(i)*PSIB(j)*PSIB(k)*PSIB(l);

                dtA = eval(int(funcA*prob,-inf,inf));
                dtB = eval(int(funcB*prob,-inf,inf));

                Dtensor{k,l}(i,j) = sparse(dtA*dtB);

            end



        end

        fprintf('\n iteration j: %d \n', j )
    end

    fprintf('\n iteration i: %d \n', i )

end


%}

save('norm2D.mat','norm')

save('Ctensor2D.mat','Ctensor')
save('Dtensor2D.mat','Ctensor')





%