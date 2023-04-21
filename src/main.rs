use std::cmp::min;

use itertools::Itertools;
use rand::{seq::SliceRandom, thread_rng, Rng};
use z3::ast::*;

#[derive(Debug)]
struct Input {
    row: usize,
    cols: Vec<(usize, usize)>,
    n_people: usize,
    max_co_assign: i32,
}

fn mat<'a>(
    context: &'a z3::Context,
    name_prefix: &str,
    row: usize,
    col: usize,
) -> Vec<Vec<Int<'a>>> {
    (0..row)
        .map(|r| {
            (0..col)
                .map(|c| Int::new_const(&context, format!("{name_prefix}-{r}-{c}")))
                .collect_vec()
        })
        .collect_vec()
}

fn is_valid(input: &Input) -> bool {
    let is_row_lo_valid = input.cols.iter().map(|(c, _)| c).sum::<usize>() <= input.n_people;
    let is_row_hi_valid = input.cols.iter().map(|(_, c)| c).sum::<usize>() >= input.n_people;
    let is_cols_lo_hi_valid = input
        .cols
        .iter()
        .all(|(mn, mx)| mn * input.row <= input.n_people && mx * input.row >= input.n_people);
    is_row_lo_valid && is_row_hi_valid && is_cols_lo_hi_valid
}

fn random_input() -> Input {
    let max_retry = 1000000;
    let mut rng = thread_rng();
    for _ in 0..max_retry {
        let row = 6;
        let col = 6;
        let cols = (0..col)
            .map(|_| {
                let lo = rng.gen_range(1..6);
                let hi = min(lo + 2, 6);
                let hi = rng.gen_range(lo..=hi);
                (lo, hi)
            })
            .collect_vec();
        let n_people = rng.gen_range(6..=36);

        let input = Input {
            row,
            cols,
            n_people,
            max_co_assign: 2,
        };

        if is_valid(&input) {
            return input;
        }
    }

    panic!("failed to gen valid input");
}

fn main() {
    let config = z3::Config::new();
    let context = z3::Context::new(&config);
    let solver = z3::Solver::new(&context);

    macro_rules! z_i64 {
        ($i:expr) => {
            z3::ast::Int::from_i64(&context, $i)
        };
    }

    // let input = Input {
    //     row: 6,
    //     cols: vec![(5, 6), (4, 6), (4, 6), (4, 6), (2, 5), (2, 5)],
    //     n_people: 30,
    //     max_co_assign: 2,
    // };

    let input = random_input();
    eprintln!("{input:?}");

    let names = (0..input.n_people).map(|i| format!("P{i}")).collect_vec();
    let ms = names
        .iter()
        .map(|name| mat(&context, name, input.row, input.cols.len()))
        .collect_vec();

    // for each matrix (i.e. staff)
    for m in &ms {
        for row in m {
            solver.assert(
                &Int::add(&context, &row.iter().collect_vec())
                    ._eq(&z_i64!(1))
                    .simplify(),
            );
        }

        for c in 0..input.cols.len() {
            let col_sum =
                Int::add(&context, &(0..input.row).map(|r| &m[r][c]).collect_vec()).simplify();
            solver.assert(&col_sum._eq(&z_i64!(1)));
        }

        for r in 0..input.row {
            for c in 0..input.cols.len() {
                let ge_zero = m[r][c].ge(&z_i64!(0));
                solver.assert(&ge_zero);
            }
        }

        if input.max_co_assign > 0 {
            for (ma, mb) in ms
                .iter()
                .combinations(2)
                .map(|c| c.into_iter().collect_tuple().unwrap())
            {
                let mut conds = vec![];
                conds.reserve(input.row * input.cols.len());
                for r in 0..input.row {
                    for c in 0..input.cols.len() {
                        let co_assigned = (&ma[r][c].gt(&z_i64!(0))) & (&mb[r][c].gt(&z_i64!(0)));
                        conds.push((co_assigned, 1));
                    }
                }
                solver.assert(
                    &Bool::pb_le(
                        &context,
                        &conds.iter().map(|(c, i)| (c, *i)).collect_vec(),
                        input.max_co_assign,
                    )
                    .simplify(),
                );
            }
        }
    }

    // for each grid
    for r in 0..input.row {
        for c in 0..input.cols.len() {
            let assigned_cnt = &Int::add(&context, &ms.iter().map(|m| &m[r][c]).collect_vec());
            let (mn, mx) = input.cols[c];
            solver.assert(
                &(assigned_cnt.ge(&z_i64!(mn as i64)) & assigned_cnt.le(&z_i64!(mx as i64))),
            );
        }
    }

    let step = 1;
    for _ in 0..step {
        let res = solver.check();
        if !matches!(res, z3::SatResult::Sat) {
            eprintln!("Unable to find solution: {:?}", res);
            break;
        }

        let model = solver.get_model().unwrap();
        let mut schedule = vec![vec![Vec::<&str>::new(); input.cols.len()]; input.row];
        for (name, m) in names.iter().zip(ms.iter()) {
            for r in 0..input.row {
                for c in 0..input.cols.len() {
                    let assigned = model.eval(&m[r][c], true).unwrap().as_i64().unwrap() == 1;
                    if assigned {
                        schedule[r][c].push(name);
                    }
                }
            }
        }

        // let weights = calculate_weights(&schedule);
        // eprintln!("Weights:");
        // for row in &weights {
        //     eprintln!("{row:?}");
        // }

        // eprintln!("Staff assigned to rooms:");
        // for (name, m) in names.iter().zip(ms.iter()) {
        //     let mut count = vec![0; input.cols.len()];
        //     for c in 0..input.cols.len() {
        //         for r in 0..input.row {
        //             let assigned = model.eval(&m[r][c], true).unwrap().as_i64().unwrap();
        //             count[c] += assigned;
        //         }
        //     }
        //     eprintln!("{name:4}: {count:?}");
        // }

        // eprintln!("Final schedule:");
        let num_w = names.iter().map(|n| n.len()).max().unwrap();
        let max_n = schedule
            .iter()
            .flat_map(|r| r.iter().map(|x| x.len()))
            .max()
            .unwrap();
        let str_w = (max_n - 1) * 2 + max_n * num_w;
        for row in &schedule {
            for s in row {
                let s = s.iter().map(|x| format!("{x:>num_w$}")).join(", ");
                print!("[ {s:>str_w$}] ");
            }

            println!("");
        }

        let mut rng = thread_rng();
        let r = rng.gen_range(0..input.row);
        let c = rng.gen_range(0..input.cols.len());
        let name = schedule[r][c].choose(&mut rng).unwrap();
        let mi = name.strip_prefix("P").unwrap().parse::<usize>().unwrap();
        solver.assert(&ms[mi][r][c]._eq(&z_i64!(0)));

        println!("");
    }
}

#[allow(dead_code)]
fn calculate_weights(result: &Vec<Vec<Vec<&str>>>) -> Vec<Vec<usize>> {
    let n = result[0].iter().map(|s| s.len()).sum();
    let mut weights: Vec<_> = (0..n).into_iter().map(|_| vec![0usize; n]).collect();

    for row in result {
        for s in row {
            for v in s.iter().combinations(2) {
                let (&&a, &&b) = v
                    .iter()
                    .collect_tuple()
                    .expect("combinations(2) should return 2-tuple");
                let a = a.strip_prefix("P").unwrap().parse::<usize>().unwrap();
                let b = b.strip_prefix("P").unwrap().parse::<usize>().unwrap();
                weights[a][b] += 1;
                weights[b][a] += 1;
            }
        }
    }

    weights
}
