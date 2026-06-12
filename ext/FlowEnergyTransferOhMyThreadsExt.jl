module FlowEnergyTransferOhMyThreadsExt

using OhMyThreads: OhMyThreads
using LinearAlgebra: LinearAlgebra
using FlowEnergyTransfer: FlowEnergyTransfer as FET
using FlowEnergyTransfer.Types: AbstractShellBinning, LinearBinning, ShellToShellResult
using FlowEnergyTransfer.ShellBinning: shell_edges, shell_centers, shell_mask
using FlowEnergyTransfer.Utils: wavenumber_magnitude_grid

# ---------------------------------------------------------------------------
# Thread-parallel shell-to-shell transfer (OhMyThreads scheduler)
# ---------------------------------------------------------------------------

"""
    threaded_shell_to_shell_transfer(velocity_hat, ks;
        binning, dealiasing=true, verify_antisymmetry=true)
        -> ShellToShellResult

Thread-parallel version of shell-to-shell transfer using OhMyThreads.
The outer loop over mediator shells (m index) is parallelised.

Requires FFTW to also be loaded for the FFT transforms.
"""
function FET.ShellToShellTransfer._shell_to_shell_threaded(
    velocity_hat::AbstractArray{<:Complex},
    ks::Tuple;
    binning::AbstractShellBinning = LinearBinning(1.0),
    dealiasing::Bool = true,
    verify_antisymmetry::Bool = true,
)
    # Delegate to the FFTW path if available, but parallelise the shell loop.
    # If FFTW ext is not loaded this will fall back to the direct path.
    nd   = length(ks)
    ns   = size(velocity_hat)[1:nd]
    D    = size(velocity_hat, nd+1)
    FT   = eltype(real(velocity_hat[1]))

    k_mag   = wavenumber_magnitude_grid(ks)
    k_max   = maximum(k_mag)
    edges   = shell_edges(binning, k_max)
    N_sh    = length(edges) - 1
    centers = shell_centers(binning, k_max)
    masks   = [shell_mask(k_mag, edges, n) for n in 1:N_sh]

    T_mat_rows = Vector{Vector{FT}}(undef, N_sh)

    # Thread-parallel loop over mediator shells
    OhMyThreads.@tasks for m in 1:N_sh
        T_mat_rows[m] = _compute_row_for_mediator(
            m, velocity_hat, masks, ks, N_sh, D, nd, FT; dealiasing=dealiasing)
    end

    T_mat = hcat(T_mat_rows...)'  # (N_sh, N_sh): row = receiver, col = mediator

    net_transfer = vec(sum(T_mat; dims=2))
    max_asym = verify_antisymmetry ? maximum(abs, T_mat .+ T_mat') : FT(NaN)

    return ShellToShellResult{FT}(
        convert(Vector{FT}, centers),
        convert(Vector{FT}, edges),
        T_mat,
        convert(Vector{FT}, net_transfer),
        max_asym,
    )
end

# Compute T[:,m] — one column of the transfer matrix for mediator shell m.
# Each task is independent so this is trivially parallel.
function _compute_row_for_mediator(
    m::Int,
    velocity_hat::AbstractArray{<:Complex},
    masks::Vector,
    ks::Tuple,
    N_sh::Int,
    D::Int,
    nd::Int,
    FT::Type;
    dealiasing::Bool,
)
    ns = size(velocity_hat)[1:nd]

    # Call the direct nonlinear term for mediator m — thread-safe (no shared state)
    û_m = zeros(Complex{FT}, size(velocity_hat)...)
    for I in CartesianIndices(ns)
        masks[m][I] || continue
        for c in 1:D
            û_m[I, c] = velocity_hat[I, c]
        end
    end

    N̂_m = FET.NonlinearTerm.compute_nonlinear_term(û_m, ks;
            dealiasing=dealiasing, backend=FET.SerialBackend())

    col = zeros(FT, N_sh)
    for n in 1:N_sh
        s = zero(FT)
        for I in CartesianIndices(ns)
            masks[n][I] || continue
            for c in 1:D
                s += real(conj(velocity_hat[I, c]) * N̂_m[I, c])
            end
        end
        col[n] = s
    end
    return col
end

# Register stub override so ThreadedBackend dispatches here
function FET.ShellToShellTransfer._shell_to_shell_threaded(args...; kwargs...)
    throw(ArgumentError("ThreadedBackend shell-to-shell requires OhMyThreads. Run `using OhMyThreads`."))
end

# ---------------------------------------------------------------------------
# Override TriadicOrthogonalDecomposition._triadic_loop_threaded!
# ---------------------------------------------------------------------------

"""
    _triadic_loop_threaded!(...)

Thread-parallel triad loop using OhMyThreads. Each triad is independent
(read-only Q_hat, writes to separate Dict slots), so this is embarrassingly parallel.
"""
function FET.TriadicOrthogonalDecomposition._triadic_loop_threaded!(
    L, P, T_budget, A_out, Xi_out,
    Q_hat, f_idx, fk_idx, fl_idx, fn_idx,
    weights, nBlks, nFreq, nState, nx, nmode,
    Q_nonlinear, LHS,
    return_coefficients, return_auxiliary_modes,
)
    nTriads = length(fk_idx)
    nStateNx = nState * nx

    # Thread-local result accumulators
    local_Ls = [fill(NaN, nFreq, nFreq, nmode) for _ in 1:Threads.nthreads()]
    local_Ts = [fill(NaN, nFreq, nFreq, nmode) for _ in 1:Threads.nthreads()]
    local_Ps = [Dict{Tuple{Int,Int}, NamedTuple}() for _ in 1:Threads.nthreads()]
    local_As = return_coefficients ? [Dict{Tuple{Int,Int}, NamedTuple}() for _ in 1:Threads.nthreads()] : nothing
    local_Xis = return_auxiliary_modes ? [Dict{Tuple{Int,Int}, NamedTuple}() for _ in 1:Threads.nthreads()] : nothing

    OhMyThreads.@tasks for i in 1:nTriads
        tid = Threads.threadid()
        fi_k = fk_idx[i]
        fi_l = fl_idx[i]
        fi_n = fn_idx[i]

        Q_n_raw = Q_hat[fi_n, :, :, :]
        Q_k_raw = Q_hat[fi_k, :, :, :]
        Q_l_raw = Q_hat[fi_l, :, :, :]

        Q_hat_n = reshape(permutedims(LHS(Q_n_raw), (2, 1, 3)), nStateNx, nBlks)
        Q_hat_kl = reshape(Q_nonlinear(Q_k_raw, Q_l_raw), nStateNx, nBlks)

        U, s, V = FET.TriadicOrthogonalDecomposition.triadic_svd(Q_hat_n, Q_hat_kl, weights, nBlks)

        nm = min(nmode, length(s))
        u = U[:, 1:nm]
        v = V[:, 1:nm]

        for j in 1:nm
            local_Ls[tid][fi_l, fi_n, j] = s[j]
            local_Ts[tid][fi_l, fi_n, j] = s[j] * real(LinearAlgebra.dot(v[:, j], weights .* u[:, j]))
        end

        local_Ps[tid][(fi_l, fi_n)] = (convective=u, recipient=v)

        if return_coefficients
            A_conv = u' * (Q_hat_kl .* weights)
            A_recip = v' * (Q_hat_n .* weights)
            local_As[tid][(fi_l, fi_n)] = (convective=A_conv, recipient=A_recip)

            if return_auxiliary_modes
                Q_hat_l = reshape(permutedims(LHS(Q_l_raw), (2, 1, 3)), nStateNx, nBlks)
                Q_hat_k = reshape(permutedims(LHS(Q_k_raw), (2, 1, 3)), nStateNx, nBlks)
                inv_s = 1 ./ s[1:nm]
                donor_mode = Q_hat_l * A_recip' * LinearAlgebra.Diagonal(inv_s) ./ nBlks
                catalyst_mode = Q_hat_k * A_recip' * LinearAlgebra.Diagonal(inv_s) ./ nBlks
                local_Xis[tid][(fi_l, fi_n)] = (donor=donor_mode[:, 1:nm], catalyst=catalyst_mode[:, 1:nm])
            end
        end
    end

    # Merge thread-local results
    for tid in 1:Threads.nthreads()
        for idx in CartesianIndices(L)
            if !isnan(local_Ls[tid][idx])
                L[idx] = local_Ls[tid][idx]
                T_budget[idx] = local_Ts[tid][idx]
            end
        end
        merge!(P, local_Ps[tid])
        if return_coefficients
            merge!(A_out, local_As[tid])
            if return_auxiliary_modes
                merge!(Xi_out, local_Xis[tid])
            end
        end
    end
end

end # module FlowEnergyTransferOhMyThreadsExt
