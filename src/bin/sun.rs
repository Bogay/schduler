use itertools::Itertools;
use rand::prelude::*;
use std::{cmp::min, collections::HashSet};

#[derive(Debug)]
struct Input {
    row: usize,
    cols: Vec<(usize, usize)>,
    n_people: usize,
    max_co_assign: i32,
}

fn is_valid(input: &Input) -> bool {
    input.cols.iter().all(|(a, b)| a == b)
}

fn random_input() -> Input {
    let max_retry = 1000000;
    for _ in 0..max_retry {
        let row = 6;
        let col = 6;
        let cols = vec![(6, 6); col];
        let n_people = cols[0].1 * col;

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

fn gen_offsets(m: usize, g: usize) -> Vec<usize> {
    let max_retry = min((1..=m).product(), 100);
    let mut possible_offsets = HashSet::new();
    let mut rng = thread_rng();
    while possible_offsets.len() < max_retry {
        let mut o = (0..m).collect_vec();
        o.shuffle(&mut rng);
        possible_offsets.insert(o);
    }

    for offsets in possible_offsets.iter().combinations(g) {
        let is_valid = (0..m).all(|i| (0..g).map(|j| offsets[j][i]).unique().count() == g);
        if is_valid {
            return offsets.into_iter().flat_map(|v| v.clone()).collect();
        }
    }

    panic!("failed to gen offsets");
}

fn solve(input: &Input) -> Vec<Vec<Vec<usize>>> {
    let offsets = gen_offsets(input.cols.len(), input.cols[0].1);
    eprintln!("Offsets:");
    for g_offset in offsets.chunks(input.cols.len()) {
        eprintln!("{g_offset:?}");
    }
    let mut secs = vec![];

    for r in 0..input.row {
        let mut sec = vec![vec![]; input.cols.len()];

        for (gi, group) in (0..input.n_people)
            .chunks(input.cols.len())
            .into_iter()
            .enumerate()
        {
            let offset = offsets[gi * input.cols.len() + r];
            for (i, p) in group.enumerate() {
                let i = (i + offset) % sec.len();
                sec[i].push(p);
            }
        }

        secs.push(sec);
    }

    secs
}

#[allow(dead_code)]
fn calculate_weights(result: &Vec<Vec<Vec<usize>>>) -> Vec<Vec<usize>> {
    let n = result[0].iter().map(|s| s.len()).sum();
    let mut weights: Vec<_> = (0..n).into_iter().map(|_| vec![0usize; n]).collect();

    for row in result {
        for s in row {
            for v in s.iter().combinations(2) {
                let (&&a, &&b) = v
                    .iter()
                    .collect_tuple()
                    .expect("combinations(2) should return 2-tuple");
                weights[a][b] += 1;
                weights[b][a] += 1;
            }
        }
    }

    weights
}

fn main() {
    let input = random_input();
    assert!(is_valid(&input));
    let schedule = solve(&input);

    let weights = calculate_weights(&schedule);
    eprintln!("Weights:");
    for row in &weights {
        eprintln!("{row:?}");
    }

    // eprintln!("Staff assigned to rooms:");
    // for m in 0..input.n_people {
    //     let mut count = vec![0; input.cols.len()];
    //     for c in 0..input.cols.len() {
    //         for r in 0..input.row {
    //             let assigned = schedule[r][c].contains(&m) as usize;
    //             count[c] += assigned;
    //         }
    //     }
    //     eprintln!("{m:4}: {count:?}");
    // }

    eprintln!("Final schedule:");
    let num_w = (0..input.n_people)
        .map(|n| n.to_string().len())
        .max()
        .unwrap();
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
}
