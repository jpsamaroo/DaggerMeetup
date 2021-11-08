using Distributed
addprocs(["helios"]; exename="/home/jpsamaroo/bin/julia-dist", exeflags="-t 8")
projdir = pwd()
@everywhere begin
    using Pkg
    Pkg.activate($projdir)
end

@everywhere using Dagger, DataFrames

#Parallelizing codes with Distributed.jl is simple and can provide an appreciable speed-up; but for complicated problems or when scaling to large problem sizes, the APIs are somewhat lacking. Dagger.jl takes parallelism to the next level, with support for GPU execution, fault tolerance, and more. Dagger's scheduler exploits every bit of parallelism it can find, and uses all the resources you can give it.

#The Distributed standard library exposes RPC primitives (remotecall) and remote channels for coordinating and executing code on a cluster of Julia processes. When a problem is simple enough, such as a trivial map operation, the provided APIs are enough to get great performance and "pretty good" scaling. However, things change when one wants to use Distributed for something complicated, like a large data pipeline with many inputs and outputs, or a full desktop application. While one *could* build these programs with Distributed, one would quickly realize that a lot of functionality will need to be built from scratch: application-scale fault tolerance and checkpointing, heterogeneous resource utilization control, and even simple load-balancing. This isn't a fault of Distributed: it just wasn't designed as the be-all-end-all distributed computing library for Julia.

#Dagger.jl takes a different approach: it is a batteries-included distributed computing library, with a variety of useful tools built-in that makes it easy to build complicated applications that can scale to whatever kind and size of resources you have at your disposal. Dagger ships with a built-in heterogeneous scheduler, which can dispatch units of work to CPUs, GPUs, and future accelerators. Dagger has a framework for checkpointing (and restoring) intermediate results, and together with fault tolerance, allows computations to safely fail partway through, and be automatically or manually resumed later. Dagger also has primitives to build complicated, dynamic execution graphs across a cluster, so users can easily implement layers on top of Dagger that provide abstractions better matching the problem at hand.

#This talk will start with a brief introduction to Dagger: what it is, how it relates to Distributed.jl, and a brief overview of the features available. Then I will take the listeners through the building of a realistic, mildly complicated application with Dagger, showcasing how Dagger makes it easy to make the application scalable, performant, and featureful. As each feature of Dagger is used, I will also point out any important caveats or alternative approaches that the listeners should consider when building their own applications. I will wrap up the talk by showing the application running at scale, and talk briefly about the future of Dagger and how listeners can help to improve it.

# But let's not get ahead of ourselves; we should first get to know Dagger at its roots: how do we use it, how does it work, and how can we best utilize it for solving our own problems? Let's start off by seeing what APIs Dagger provides for us:

## Tables

N = 1000
df = DataFrame(a=rand(1:4, N), b=rand('a':'d', N))
dt = DTable(df, 100)
collect(dt.chunks[1])
fetch(dt)
tabletype(dt)

# Some simple ops
fetch(map(row->(;c=repr(row.a)*row.b), dt))
fetch(reduce(*, dt))
fetch(reduce(+, map(row->(;a=row.a), dt)))
fetch(filter(row->row.b == 'd', dt))

# Groupby
gdt = Dagger.groupby(dt, :b)
gdt['c']
fetch(gdt['c'])
for (key, t) in gdt
    @show key first(fetch(t))
end

# CSVs
@everywhere using CSV

# Reading directly to DTable
df |> CSV.write("test.csv")
df2 = CSV.File("test.csv") |> DataFrame
dt2 = CSV.File("test.csv") |> DTable
DataFrame(dt2)
DataFrame(dt2) == df2

# Writing directly from DTable
dt2 |> CSV.write("test2.csv")
DataFrame(CSV.File("test2.csv")) == DataFrame(dt2)

# Loading multiple CSVs
df |> CSV.write("test3.csv")
dt3 = DTable(CSV.File, ["test.csv", "test2.csv", "test3.csv"])
DataFrame(dt3) == vcat(df2, df2, df2)
tabletype!(dt3)

## Arrays (API subject to change soon)

# Allocation and computation
X = rand(Blocks(64, 64), 256, 256)
DX = compute(X)
collect(X)

# Map and reduce
collect(map(x->x+3, DX))
collect(reduce(+, DX))[1] ≈ sum(collect(DX))

# Matmul
collect(DX * DX) ≈ collect(DX) * collect(DX)

# Getindex
collect(DX[2:3,:]) == collect(DX)[2:3,:]

# Broadcasting
collect(DX .* 2) == collect(DX) .* 2
collect(DX .* DX) == collect(DX) .* collect(DX)
collect(DX .* collect(DX)) == collect(DX .* DX)

## Raw Operations

a = Dagger.@spawn 1+1
fetch(a)
b = Dagger.@spawn a*3
c = Dagger.@spawn b+4
fetch(c)

# Errors
a = Dagger.@spawn 1 + "a"
fetch(a)
b = Dagger.@spawn a+1
fetch(b)
wait(b);
@time wait(Dagger.@spawn sleep(3));

## Scheduler Control

fetch(Dagger.@spawn single=1 myid())
#fetch(Dagger.@spawn single=2 myid())

# Processors and Scopes

Threads.nthreads()
t1 = Dagger.ThreadProc(1, 1)
t2 = Dagger.ThreadProc(1, 2)
t7 = Dagger.ThreadProc(1, 7)
fetch(Dagger.@spawn scope=Dagger.ExactScope(t1) Threads.threadid())
fetch(Dagger.@spawn scope=Dagger.ExactScope(t2) Threads.threadid())
fetch(Dagger.@spawn scope=Dagger.ExactScope(t7) Threads.threadid())

## Scheduler Internals

@everywhere using DaggerWebDash
@everywhere import DaggerWebDash: GanttPlot, LinePlot

ml = Dagger.MultiEventLog()
ml[:core] = Dagger.Events.CoreMetrics()
ml[:id] = Dagger.Events.IDMetrics()
ml[:timeline] = Dagger.Events.TimelineMetrics()
ml[:wsat] = Dagger.Events.WorkerSaturation()
ml[:loadavg] = Dagger.Events.CPULoadAverages()
ml[:bytes] = Dagger.Events.BytesAllocd()
ml[:mem] = Dagger.Events.MemoryFree()
ml[:esat] = Dagger.Events.EventSaturation()
ml[:psat] = Dagger.Events.ProcessorSaturation()
lw = Dagger.Events.LogWindow(20*10^9, :core)
d3r = DaggerWebDash.D3Renderer(8080)
push!(lw.creation_handlers, d3r)
push!(lw.deletion_handlers, d3r)
push!(d3r, GanttPlot(:core, :id, :timeline, :esat, :psat, "Overview"))
push!(d3r, LinePlot(:core, :wsat, "Worker Saturation", "Running Tasks"))
push!(d3r, LinePlot(:core, :loadavg, "CPU Load Average", "Average Running Threads"))
push!(d3r, LinePlot(:core, :bytes, "Allocated Bytes", "Bytes"))
push!(d3r, LinePlot(:core, :mem, "Available Memory", "% Free"))
ml.aggregators[:logwindow] = lw
ml.aggregators[:d3r] = d3r
Dagger.Sch.eager_context().log_sink = ml
