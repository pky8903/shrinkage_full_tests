#pragma once
#include <cstddef>
#include <utility>

namespace st { namespace util {

/// The ONE interface that crosses the .so / compilation-unit boundary.
/// Platform only knows this type. User wraps their function to match it.
using AdaptedFunc = void (*)(void** args);


// ─────────────────────────────────────────────────────────────────
//  FuncAdapter
//
//  Wraps any function into the uniform AdaptedFunc signature:
//    void fn(void** args)
//
//  Usage (in user's .cc or .cu, next to their function):
//
//    constexpr AdaptedFunc funcA_adapted =
//        FuncAdapter<funcA, float*, float*, int*>::call;
//
//  Pass the _adapted name to register_pair() / register_adjoint().
//  The type list must match your ArgDescriptor list exactly.
//
//  Supports any number of arguments. Requires C++17.
//
//  C++17 feature: `template <auto Func>` — Non-Type Template Parameter
//  with deduced type (NTTP). The compiler deduces the type of Func from
//  whatever function is passed, so you don't need to spell it out.
//
//  At compile time, std::index_sequence<0,1,...,N-1> expands the pack:
//    Func(reinterpret_cast<Ts>(args[Is])...)
//    → Func((T0)args[0], (T1)args[1], ..., (TN-1)args[N-1])
//
//  NOTE: scalar parameters (int, float) must be passed as pointers
//  (int*, float*). Allocate a 1-element PARAM buffer and dereference
//  inside your wrapper function.
// ─────────────────────────────────────────────────────────────────

template <auto Func, typename... Ts>
struct FuncAdapter {
private:
    template <std::size_t... Is>
    static void call_impl(void** args, std::index_sequence<Is...>) {
        Func(reinterpret_cast<Ts>(args[Is])...);
    }
public:
    static void call(void** args) {
        call_impl(args, std::make_index_sequence<sizeof...(Ts)>{});
    }
};

} } // namespace st::util
