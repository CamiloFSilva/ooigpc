clear all

num=4;

syms x

psi0 = 1;
psi1 = x;
psi2 = x^2-1;
psi3 = x^3-3*x;


PSI = [psi0, psi1, psi2, psi3];

prob = 1/(sqrt(2*pi))*exp(-x^2/2);

%%

for k=1:num

    func = PSI(k)*PSI(k);
    norm(k) = eval(int(func*prob,-inf,inf));

end



for k=1:num
    for j=1:num
        for i=1:num

            func = PSI(i)*PSI(j)*PSI(k);

            ct = eval(int(func*prob,-inf,inf));

            Ctensor(i,j,k) = ct;

        end
    end

end


for i=1:num
    for j=1:num
        for k=1:num
            for l=1:num

                funcAC = PSI(i)*PSI(j)*PSI(k)*PSI(l);

                ctA = eval(int(funcAC*prob,-inf,inf));

                Dtensor{k,l}(i,j) = sparse(ctA);

            end

        end

        fprintf('\n iteration j: %d \n', j )
    end

    fprintf('\n iteration i: %d \n', i )

end









%